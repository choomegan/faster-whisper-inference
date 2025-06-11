import json
import logging
import os
import re
import time
from typing import Dict, List, Tuple, Union

import jiwer
import tqdm
import yaml
from faster_whisper import WhisperModel

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

##################### config['evaluation']S #########################
config = yaml.safe_load(open("config.yaml", encoding="utf-8"))

MODEL_DIR = config["evaluation"]["ct2_model_dir"]
MANIFEST_PATH = config["evaluation"]["manifest_path"]
OUTPUT_PATH = config["evaluation"]["output_path"]
BEAM_SIZE = config["evaluation"]["beam_size"]
DEVICE = config["evaluation"]["device"]
COMPUTE_TYPE = config['evaluation']['quant']
#######################################################


def compute_asr_metrics(
    reference: Union[List, str], hypothesis: Union[List, str]
) -> Tuple[float]:
    """
    Use Jiwer to calculate asr metrics
    """
    wer = round(jiwer.wer(reference, hypothesis) * 100, 3)
    cer = round(jiwer.cer(reference, hypothesis) * 100, 3)
    mer = round(jiwer.mer(reference, hypothesis) * 100, 3)

    return wer, cer, mer


def normalise(text: str) -> str:
    """
    Normlize text for fair wer comparison
    """
    text = text.lower()
    # remove unwanted chars
    cleaned = re.sub(r"[^A-Za-z0-9#\' ]+", " ", text)
    # remove multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    if "<unk>" in cleaned:
        cleaned = re.sub(r"\s*<unk>\s*", " ", cleaned)
    return cleaned.strip()


def compute_asr_metrics(
    reference: Union[List, str], hypothesis: Union[List, str]
) -> Tuple[float]:
    """
    Use Jiwer to calculate asr metrics
    """
    wer = round(jiwer.wer(reference, hypothesis) * 100, 3)
    cer = round(jiwer.cer(reference, hypothesis) * 100, 3)
    mer = round(jiwer.mer(reference, hypothesis) * 100, 3)

    return wer, cer, mer


def load_manifest_nemo(input_manifest_path: str) -> List[Dict[str, str]]:
    """
    loads the manifest file in Nvidia NeMo format to process the entries and store them
    into a list of dictionaries

    input_manifest_path: the manifest path that contains the information of the audio
    clips of interest
    ---
    returns: a list of dictionaries of the information in the input manifest file
    """

    dict_list = []

    with open(input_manifest_path, "r+", encoding="utf-8") as f:
        for line in f:
            dict_list.append(json.loads(line))

    return dict_list


def transcribe_files():
    """
    Main method to transcribe files from a manifest
    """
    logging.info("Loading model......")
    model = WhisperModel(MODEL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
    logging.info("Model loaded!")

    base_dir = os.path.dirname(MANIFEST_PATH)

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    data = load_manifest_nemo(MANIFEST_PATH)

    references = []
    hypotheses = []

    start = time.time()
    for item in tqdm.tqdm(data):
        audio_filepath = os.path.join(base_dir, item["audio_filepath"])
        segments, _ = model.transcribe(audio_filepath, beam_size=BEAM_SIZE)

        segments = list(segments)  # The transcription will actually run here.
        transcript = " ".join(
            segment.text for segment in segments
        )  # join multiple segments
        item["pred"] = transcript

        text = normalise(item["text"])
        pred_text = normalise(transcript)

        references.append(text)
        hypotheses.append(pred_text)

        wer = jiwer.wer(text, pred_text)
        item["wer"] = wer

        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )
    end = time.time()
    logging.info("Time elapsed (minutes): %s", (end - start) / 60)
    overall_wer, overall_cer, overall_mer = compute_asr_metrics(references, hypotheses)
    logging.info("WER: %s%", overall_wer)
    logging.info("CER: %s%", overall_cer)
    logging.info("Word Accuracy: %s%", overall_mer)


if __name__ == "__main__":
    transcribe_files()

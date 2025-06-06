"""
Script to convert whisper model to ctranslate2
"""

import subprocess
import logging

import yaml
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)
##################### CONFIGS #########################
config = yaml.safe_load(open("config.yaml", encoding="utf-8"))

MODEL_FILEPATH = config["convert_model"]["model_filepath"]
OUTPUT_MODEL_FILEPATH = config["convert_model"]["output_model_filepath"]
QUANTIZATION = config["convert_model"]["quant"]
#######################################################


def convert_model():
    """
    Run shell command to convert model to ct2 format
    """
    command = [
        "ct2-transformers-converter",
        "--model",
        MODEL_FILEPATH,
        "--output_dir",
        OUTPUT_MODEL_FILEPATH,
        "--quantization",
        QUANTIZATION,
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    # Always print STDOUT if available
    if result.stdout.strip():
        logging.info("STDOUT:\n %s", result.stdout)

    # Print STDERR only if the command failed or contains useful messages
    if result.returncode != 0:
        logging.error("STDERR:\n %s", result.stderr)
    elif result.stderr.strip():
        # Optionally print non-fatal warnings
        logging.info("Non-fatal STDERR:\n %s", result.stderr)

    logging.info("Saved ct2 model to: %s", OUTPUT_MODEL_FILEPATH)
    
if __name__ == "__main__":
    convert_model()

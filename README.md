This repository is a quick way to evaluate your Whisper models with [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) without having to wait a gigaJIbillion years. With beam size 5, the time savings is ~10x.


## Quick Start
Change the volume mounts in [docker-compose.yaml](docker-compose.yaml), then run:
```
docker compose up -d
docker exec -it faster-whisper-inf bash
```
Configuration file can be found [here](config.yaml).
## Converting model to cTranslate2
Edit `convert_model` portion in config file. 

Possible values for `quant` can be found here: https://opennmt.net/CTranslate2/quantization.html.
```
convert_model:
  model_filepath:
  output_model_filepath:
  quant: float32 
```
Run the following command to convert the model to cTranslate2 format.
```
python3 convert_model.py
```

## Running inference
Edit `evaluation` portion in config file. 
```
evaluation:
  ct2_model_dir:
  beam_size:
  device: # 'cuda'/'cpu'/'auto'
  manifest_path:
  output_path:
```
Then run:
```
python3 faster-inf.py
```

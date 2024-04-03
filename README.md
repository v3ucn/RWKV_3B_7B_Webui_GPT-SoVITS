## Install python3.11 locally

## Make sure that visual studio install is installed locally and the environment variables of cl.exe are configured

## Make sure the local cuda version is 11

```
pip install -r requirements.txt
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
```

## Download the model to the root directory of the project

https://huggingface.co/a686d380/rwkv-5-h-world

```
python3 webui_gpu.py
```


# Skin Cancer Classification (MobileNetV3)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --data_root "/home/semi/Vscode/Rawdata Comvis" --output_dir "/home/semi/Vscode/Comvis/models" --epochs 10 --batch_size 32 --device cpu
```

## Run App

```bash
python app_gradio.py
```

Upload a dermoscopic image and see predicted class and probabilities.

## File Structure

```
Comvis/
├── src/
│   ├── __init__.py
│   ├── utils.py
│   ├── dataset.py
│   └── train.py
├── models/
│   ├── mobilenetv3_ham10000.pt
│   └── class_mapping.json
├── app_gradio.py
├── requirements.txt
└── README.md
```
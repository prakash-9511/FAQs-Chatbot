# RCOEM Fine-Tuned Chatbot

This project fine-tunes a Hugging Face text-generation model on RCOEM FAQ-style data and uses the saved model in a small web chatbot UI.

## What It Includes

- `data/rcoem_faqs.jsonl` - clean FAQ training examples.
- `train.py` - downloads a Hugging Face causal language model and fine-tunes it.
- `app.py` - Flask chatbot server that loads the fine-tuned model from `models/rcoem-chatbot`.
- `templates/index.html` and `static/styles.css` - chatbox UI for simulating the RCOEM website chatbot.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Fine-Tune The Model

The default model is `distilgpt2`, a small Hugging Face text-generation model suitable for local demos.

```powershell
python train.py
```

The trained model is saved to:

```text
models/rcoem-chatbot
```

If Python cannot create files inside a OneDrive-synced folder, save to a writable path first:

```powershell
python train.py --output_dir "$env:TEMP\rcoem-chatbot"
$env:RCOEM_MODEL_DIR="$env:TEMP\rcoem-chatbot"
python app.py
```

You can choose another Hugging Face causal language model:

```powershell
python train.py --model_name microsoft/DialoGPT-small --output_dir models/rcoem-chatbot
```

## Run The Chatbot

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

If the fine-tuned model is not available yet, the app still works using FAQ retrieval so the UI can be tested immediately. After training, it uses the saved fine-tuned model for generation.

## Add More FAQs

Append new examples to `data/rcoem_faqs.jsonl` in this format:

```json
{"question":"Your question?","answer":"Your verified answer.","source":"Official URL or internal source"}
```

Keep answers short, unambiguous, and based on official RCOEM/RBU sources.

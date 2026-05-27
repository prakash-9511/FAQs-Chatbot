import json
import os
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "rcoem_faqs.jsonl"
MODEL_DIR = Path(os.getenv("RCOEM_MODEL_DIR", BASE_DIR / "models" / "rcoem-chatbot"))

app = Flask(__name__)
faq_items = []
tokenizer = None
model = None
torch = None
device = "cpu"


def normalize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_faqs():
    with DATA_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def best_faq_match(question):
    query_terms = normalize(question)
    if not query_terms:
        return None, 0.0

    best_item = None
    best_score = 0.0
    for item in faq_items:
        terms = normalize(item["question"] + " " + item["answer"])
        overlap = len(query_terms & terms)
        score = overlap / max(len(query_terms), 1)
        if score > best_score:
            best_item = item
            best_score = score
    return best_item, best_score


def load_model():
    global tokenizer, model, torch, device

    if not (MODEL_DIR / "config.json").exists():
        return

    try:
        import torch as torch_module
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        return

    torch = torch_module
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()


def generate_answer(question, faq_context):
    if model is None or tokenizer is None:
        return faq_context["answer"], "faq-retrieval"

    prompt = (
        "You are the official-style RCOEM website chatbot. "
        "Answer only from verified RCOEM FAQ data.\n\n"
        f"Verified FAQ answer: {faq_context['answer']}\n"
        f"Question: {question}\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=90,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    answer = text.split("Answer:", 1)[-1].strip()

    faq_terms = normalize(faq_context["answer"])
    answer_terms = normalize(answer)
    overlap = len(faq_terms & answer_terms) / max(len(faq_terms), 1)
    if not answer or len(answer.split()) < 4 or overlap < 0.18:
        answer = faq_context["answer"]
    return answer, "fine-tuned-model-with-faq-grounding"


@app.route("/")
def index():
    return render_template("index.html", model_ready=model is not None)


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()

    if not question:
        return jsonify({"answer": "Please enter a question about RCOEM.", "source": "validation"})

    faq_context, score = best_faq_match(question)
    if faq_context is None or score < 0.18:
        return jsonify(
            {
                "answer": "I do not have verified information for that question in my RCOEM FAQ data. Please check the official RCOEM website or contact the institute for the latest details.",
                "source": "no-match",
            }
        )

    answer, source_type = generate_answer(question, faq_context)
    return jsonify(
        {
            "answer": answer,
            "source": source_type,
            "reference": faq_context.get("source", ""),
        }
    )


if __name__ == "__main__":
    faq_items = load_faqs()
    load_model()
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=True)

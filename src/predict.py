import json
import random
import torch
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score

# Crawl text từ URL
def fetch_plain_text(url, max_len=5000):
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "footer", "nav", "iframe", "form", "button"]):
            tag.extract()
        text = " ".join(soup.get_text().split())
        return (url + " " + text)[:max_len]
    except:
        return url

# Predict nhãn cho list URL
def predict_urls(urls, model_path="models/saved_model"):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    texts = [fetch_plain_text(u) for u in urls]
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds_ids = torch.argmax(outputs.logits, dim=1).cpu().numpy()

    label2id = {"chính trị": 0, "cờ bạc": 1, "18+": 2, "nội dung khác": 3}
    id2label = {v: k for k, v in label2id.items()}
    pred_labels = [id2label[i] for i in preds_ids]
    return pred_labels

if __name__ == "__main__":
    with open("data/labeled/dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Lấy ngẫu nhiên 100 mẫu test
    sample_data = random.sample(data, min(100, len(data)))
    urls = [item["url"] for item in sample_data]
    true_labels = [item["label"] for item in sample_data]

    pred_labels = predict_urls(urls)

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}\n")

    for u, t, p in zip(urls, true_labels, pred_labels):
        print(f"URL: {u}\nTrue: {t} | Predicted: {p}\n")
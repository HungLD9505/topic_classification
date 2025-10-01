import json
import pandas as pd
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Label mappings
label2id = {"chính trị": 0, "cờ bạc": 1, "18+": 2, "nội dung khác": 3}
id2label = {v: k for k, v in label2id.items()}


def load_dataset(json_path="data/labeled/dataset.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build texts and labels from dataset.json entries
    texts = []
    labels = []
    for item in data:
        text_part = item.get("text", "")
        kw_part = " ".join(item.get("keywords_found", [])) if isinstance(item.get("keywords_found", []), list) else str(item.get("keywords_found", ""))
        combined_text = f"{item.get('url','')} {text_part} {kw_part}"
        texts.append(combined_text)
        lbl = item.get("label", "nội dung khác")
        # fallback if unknown label
        labels.append(label2id.get(lbl, label2id["nội dung khác"]))
    return texts, labels


def prepare_data(texts, labels, test_size=0.2, random_state=42):
    # Stratified split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Oversampling minority classes in train set
    train_df = pd.DataFrame({"text": train_texts, "label": train_labels})
    train_counts = train_df["label"].value_counts()
    max_count = train_counts.max()

    balanced_dfs = []
    for lbl in train_counts.index:
        df_class = train_df[train_df["label"] == lbl]
        df_resampled = resample(df_class, replace=True, n_samples=max_count, random_state=random_state)
        balanced_dfs.append(df_resampled)

    train_df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df_balanced["text"].tolist(), train_df_balanced["label"].tolist(), val_texts, val_labels


class TopicDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


def train_model(json_path="data/labeled/dataset.json",
                output_model_dir="data/saved_model",
                num_epochs=3,
                per_device_batch_size=16,
                learning_rate=5e-5,
                max_length=512):

    texts, labels = load_dataset(json_path)
    train_texts, train_labels, val_texts, val_labels = prepare_data(texts, labels)

    # Tokenizer & encodings
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)

    train_dataset = TopicDataset(train_encodings, train_labels)
    val_dataset = TopicDataset(val_encodings, val_labels)

    # Class weights (inverse frequency)
    orig_counts = Counter(train_labels)
    weights = [1.0 / orig_counts.get(i, 1) for i in range(len(label2id))]
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    # Custom Trainer to use class weights in loss
    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Save model + tokenizer
    trainer.model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Model and tokenizer saved to {output_model_dir}")

    # Evaluate on validation set
    preds = trainer.predict(val_dataset)
    y_true = val_labels
    y_pred = np.argmax(preds.predictions, axis=1)

    # Confusion matrix + classification report
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(label2id.keys()))
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix on Validation Set")
    plt.show()

    print("\n------ Classification Report ------")
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys()), digits=4))


if __name__ == "__main__":
    train_model()
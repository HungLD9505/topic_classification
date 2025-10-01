from src.crawl_label import load_urls_from_csv, crawl_all, save_to_json
from src.train import load_dataset, prepare_data, train_model
from src.predict import predict_urls

def main():
    # Crawl and label
    csv_path = "data/raw/raw_data.csv"
    urls_with_labels = load_urls_from_csv(csv_path)
    dataset = crawl_all(urls_with_labels, batch_size=5000, max_workers=200)
    save_to_json(dataset)
    print(f"Crawl xong, lưu {len(dataset)} mẫu vào dataset.json")

    # Train
    texts, labels = load_dataset()
    train_texts, train_labels, val_texts, val_labels = prepare_data(texts, labels)
    train_model(train_texts, train_labels, val_texts, val_labels)

    # Predict
    sample_data = random.sample(dataset, min(100, len(dataset)))
    urls = [item["url"] for item in sample_data]
    pred_labels = predict_urls(urls)
    print(f"Predicted {len(pred_labels)} URLs")

if __name__ == "__main__":
    main()
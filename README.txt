Topic Classification (URL)

Phân loại URL thành 4 nhóm: chính trị, cờ bạc, 18+, nội dung khác bằng mô hình DistilBERT.

---

Yêu cầu

- Python 3.10+
- pip

---

Cài đặt

git clone https://github.com/HungLD9505/topic_classification.git
cd topic-classification

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt

---

Cấu trúc chính

data/            # dữ liệu
notebook/        # file thử nghiệm
src/             # code train / predict / crawl

---

Chạy code

1. Crawl dữ liệu

python src/crawl_label.py

2. Train model

python src/train.py

3. Dự đoán

python src/predict.py
---

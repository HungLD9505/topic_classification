# Topic Classification (URL)

Phân loại URL thành 4 nhóm: **chính trị, cờ bạc, 18+, nội dung khác** bằng mô hình DistilBERT.

---

## Yêu cầu

* Python 3.10+
* pip

---

## Cài đặt

```bash
git clone https://github.com/HungLD9505/topic-classification.git
cd topic-classification

python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

---

## Cấu trúc chính

```
data/            # dữ liệu
notebook/        # file thử nghiệm
src/             # code train / predict / crawl
```

```markdown
Chuẩn bị dữ liệu gốc

Hiện tại repo chỉ chứa **dữ liệu đã gán nhãn (labeled data)**.  
Để chạy pipeline hoặc huấn luyện model, bạn cần chuẩn bị dữ liệu gốc (raw data) theo cấu trúc sau:

```

data/
└── raw/
└── raw_data.csv

```

- `raw_data.csv` phải có cột **`checked_url`**.  
- Cột `checked_url` được sử dụng để crawl dữ liệu từ URL, chuẩn bị cho bước train model.

Ví dụ nội dung `raw_data.csv`

| checked_url           |
|----------------------|
| https://example.com/1 |
| https://example.com/2 |

> Lưu ý: Đặt file `raw_data.csv` vào đúng thư mục `data/raw/` để các script có thể đọc và xử lý tự động.
```

---

## Chạy

### 1. Crawl dữ liệu

```bash
python src/crawl_label.py
```

### 2. Train model

```bash
python src/train.py
```

### 3. Dự đoán

```bash
python src/predict.py
```
---

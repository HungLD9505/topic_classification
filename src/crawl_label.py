import pandas as pd
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import json
import warnings

# Keywords dictionary
keywords_dict = {
    "chính trị": ["quochoi","gov","chinhphu","nghiquyet","election","baocu","dang","party","canbo","politics",
                  "quốc hội","chính phủ","nghị quyết","bầu cử","dân chủ","vote","ứng cử","cơ quan nhà nước",
                  "luật","chính sách", "đảng","cán bộ","chính trị"],
    "cờ bạc": ["casino","bet","xoso","nhacai","slot","poker","baccarat","jackpot","danhbac","wager",
               "cá cược","xổ số","nhà cái","đánh bạc","quay thưởng","vòng quay","trò chơi","đặt cược",
               "giải đấu","bonus", "uy tín", "game", "bài"],
    "18+": ["phim18","jav","sex","nguoilon","erotic","porn","18+","adult","hentai","sexvideo",
            "phim người lớn","xxx","khiêu dâm","video sex","người lớn","nude","quan hệ","sexmovie","adultfilm","pornstar"]
}

def load_urls_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    urls = [str(url).strip() for url in df["checked_url"]]
    return [{"checked_url": url} for url in urls]

def fetch_content(url):
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        warnings.filterwarnings("ignore")
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "noscript", "footer", "nav", "form", "button", "iframe"]):
            tag.extract()
        text = " ".join(soup.get_text().split())
        if len(text) < 30 or any(err in text.lower() for err in ["error", "not found", "denied"]):
            return None
        return text[:5000]
    except Exception as e:
        with open("error_urls.txt", "a", encoding="utf-8") as f:
            f.write(f"{url} | {str(e)}\n")
        return None

def assign_label(url, text, keywords_dict):
    url_lower = url.lower()
    text_lower = text.lower()
    match_count = {topic: sum(1 for kw in kws if kw.lower() in url_lower or kw.lower() in text_lower)
                   for topic, kws in keywords_dict.items()}
    max_count = max(match_count.values())
    if max_count == 0:
        return "nội dung khác"
    candidates = [topic for topic, count in match_count.items() if count == max_count]
    return candidates[0]

def crawl_and_label(urls_with_labels, max_workers=50):
    results = []
    def worker(item):
        text = fetch_content(item["checked_url"])
        if text:
            label = assign_label(item["checked_url"], text, keywords_dict)
            keywords_found = [kw for kw in sum(keywords_dict.values(), []) if kw.lower() in text.lower() or kw.lower() in item["checked_url"].lower()]
            return {
                "url": item["checked_url"],
                "label": label,
                "text": text,
                "keywords_found": list(set(keywords_found))
            }
        return None
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker, item) for item in urls_with_labels]
        results = [future.result() for future in futures if future.result()]
    return results

def crawl_all(urls_with_labels, batch_size=5000, max_workers=200):
    all_results = []
    for i in range(0, len(urls_with_labels), batch_size):
        batch = urls_with_labels[i:i+batch_size]
        print(f"Crawl batch {i//batch_size + 1}: {len(batch)} URLs")
        batch_results = crawl_and_label(batch, max_workers=max_workers)
        all_results.extend(batch_results)
    return all_results

def save_to_json(data, filename="data/dataset.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    csv_path = "data/raw/raw_data.csv"
    urls_with_labels = load_urls_from_csv(csv_path)
    dataset = crawl_all(urls_with_labels, batch_size=5000, max_workers=200)
    save_to_json(dataset)
    print(f"Crawl xong, lưu {len(dataset)} mẫu vào dataset.json")
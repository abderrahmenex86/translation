import os
import random
import urllib.request
import zipfile


def download_and_prepare_data():
    url = "http://www.manythings.org/anki/fra-eng.zip"
    zip_path = "fra-eng.zip"
    extract_dir = "dataset/tatoeba"

    os.makedirs(extract_dir, exist_ok=True)

    print("-> Downloading french-english dataset.")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
        while chunk := response.read(8192):
            out_file.write(chunk)

    print("-> Extracting.")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    txt_file = os.path.join(extract_dir, "fra.txt")
    print("-> Processing and splitting data.")

    pairs = []
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))

    random.seed(42)
    random.shuffle(pairs)

    val_size = 10000
    train_pairs = pairs[val_size:]
    val_pairs = pairs[:val_size]

    print(f"Total pairs: {len(pairs)}")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")

    def save_split(split_pairs, prefix):
        with open(os.path.join(extract_dir, f"{prefix}.en"), "w", encoding="utf-8") as f_en, open(
            os.path.join(extract_dir, f"{prefix}.fr"), "w", encoding="utf-8"
        ) as f_fr:
            for en, fr in split_pairs:
                f_en.write(en + "\n")
                f_fr.write(fr + "\n")

    save_split(train_pairs, "train")
    save_split(val_pairs, "val")

    os.remove(zip_path)
    print(f"-< Done. Files saved in '{extract_dir}'")


if __name__ == "__main__":
    download_and_prepare_data()

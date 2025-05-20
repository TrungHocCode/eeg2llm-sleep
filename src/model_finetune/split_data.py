import json
import random


def split_dataset():
    """
    Chia dữ liệu từ dataset.jsonl thành các tập train, validation và test.
    """
    # Load dữ liệu
    with open("dataset.jsonl", "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Xáo trộn ngẫu nhiên (rất quan trọng!)
    random.seed(42)
    random.shuffle(data)

    # Tính số lượng mẫu
    total = len(data)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)
    test_size = total - train_size - val_size

    # Chia tập
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Ghi ra 3 file
    save_jsonl("train.jsonl", train_data)
    save_jsonl("val.jsonl", val_data)
    save_jsonl("test.jsonl", test_data)

    print(f"Done! Tổng {total} mẫu đã chia thành:")
    print(f"  - Train: {len(train_data)}")
    print(f"  - Validation: {len(val_data)}")
    print(f"  - Test: {len(test_data)}")


def save_jsonl(filename, dataset):
    """
    Lưu dữ liệu dạng JSONL vào file.
    
    Args:
        filename: Tên file đầu ra
        dataset: Dữ liệu cần lưu
    """
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def format_data(in_file, out_file):
    """
    Định dạng lại dữ liệu từ file JSONL thành định dạng phù hợp cho GPT-2.
    
    Args:
        in_file: File đầu vào (JSONL)
        out_file: File đầu ra (JSON)
    """
    with open(in_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    formatted = []
    for item in lines:
        prompt = item["input"].strip()
        completion = item["output"].strip()
        text = f"{prompt}\n<|sep|>\n{completion}"
        formatted.append({"text": text})

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)


def format_all_datasets():
    """
    Định dạng lại tất cả các tập dữ liệu (train, val, test).
    """
    # Gọi hàm cho cả 3 tập
    format_data("train.jsonl", "train_gpt2.json")
    format_data("val.jsonl", "val_gpt2.json")
    format_data("test.jsonl", "test_gpt2.json")
    
    print("Đã định dạng dữ liệu cho cả 3 tập train, validation và test.")
if __name__ == "__main__":
    split_dataset()
    format_all_datasets()
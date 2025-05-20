import json
from collections import defaultdict


def prepare_dataset():
    """
    Chuẩn bị dữ liệu từ file input_output.txt, phân loại và cân bằng các loại chẩn đoán.
    """
    # Tập để gom theo loại chẩn đoán
    diagnosis_map = defaultdict(list)
    all_data = []

    # Đọc file dữ liệu đầu vào
    with open('input_output.txt', 'r', encoding='utf-8') as f:
        content = f.read()

    # Tách thành từng cặp input-output
    pairs = content.split('---\n')
    for pair in pairs:
        if not pair.strip():
            continue
        try:
            input_part = pair.split('Input:\n')[1].split('Output:\n')[0].strip()
            output_part = pair.split('Output:\n')[1].strip()
        except IndexError:
            continue  # skip malformed pair

        record = {'input': input_part, 'output': output_part}
        all_data.append(record)

        # Trích xuất dòng Diagnose:
        diagnose_line = next((line for line in input_part.split('\n') if 'Diagnose:' in line), "").lower()
        diagnosis = diagnose_line.replace('diagnose:', '').strip()

        # Phân loại
        if "cerebral infarction" in diagnosis:
            diagnosis_map["cerebral infarction"].append(record)
        elif "seizure" in diagnosis or "epilepsy" in diagnosis:
            diagnosis_map["seizure"].append(record)
        elif "sleep" in diagnosis:
            diagnosis_map["sleep disorders"].append(record)
        elif "normal" in diagnosis:
            diagnosis_map["normal"].append(record)
        else:
            diagnosis_map["unknown"].append(record)

    # Đếm số lượng
    counts = {k: len(v) for k, v in diagnosis_map.items()}
    print("📊 Số lượng mẫu ban đầu:")
    for k, v in counts.items():
        print(f"  - {k}: {v}")

    # Cân bằng số lượng bằng nhân bản
    max_count = max(counts[k] for k in ["cerebral infarction", "seizure", "sleep disorders", "normal"])

    balanced_data = []
    for key in ["cerebral infarction", "seizure", "sleep disorders", "normal"]:
        records = diagnosis_map[key]
        if not records:
            continue
        times = max_count // len(records)

        balanced = records * times * 4
        balanced_data.extend(balanced)

    print(f"\n✅ Tổng số mẫu sau cân bằng: {len(balanced_data)}")

    # Ghi ra file
    with open('dataset.jsonl', 'w', encoding='utf-8') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    prepare_dataset()
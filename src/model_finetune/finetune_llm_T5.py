# !pip install transformers datasets
# !pip install peft accelerate bitsandbytes
import json, random
from collections import defaultdict

# Đọc dữ liệu gốc
with open("input_output.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Tách từng cặp input/output
pairs = content.split('---\n')
records = []
diagnosis_map = defaultdict(list)

for pair in pairs:
    if not pair.strip():
        continue
    try:
        input_part = pair.split('Input:\n')[1].split('Output:\n')[0].strip()
        output_part = pair.split('Output:\n')[1].strip()
    except IndexError:
        continue

    sample = {
        "input": input_part,
        "target": output_part
    }
    records.append(sample)

    # Trích loại bệnh
    diagnose_line = next((line for line in input_part.split('\n') if 'Diagnose:' in line), "").lower()
    diagnosis = diagnose_line.replace('diagnose:', '').strip()

    if "cerebral infarction" in diagnosis:
        diagnosis_map["cerebral infarction"].append(sample)
    elif "seizure" in diagnosis or "epilepsy" in diagnosis:
        diagnosis_map["seizure"].append(sample)
    elif "sleep" in diagnosis:
        diagnosis_map["sleep disorders"].append(sample)
    elif "normal" in diagnosis:
        diagnosis_map["normal"].append(sample)
    else:
        diagnosis_map["unknown"].append(sample)
# Cân bằng thành 4 nhóm đều 1753 mẫu
desired_total = 7012
desired_per_group = desired_total // 4
balanced_data = []

for key in ["cerebral infarction", "seizure", "sleep disorders", "normal"]:
    samples = diagnosis_map[key]
    if not samples:
        continue
    times = desired_per_group // len(samples)
    extra = desired_per_group % len(samples)
    balanced = samples * times + samples[:extra]
    balanced_data.extend(balanced)

print(f"Tổng số mẫu sau cân bằng: {len(balanced_data)}")
# Shuffle
random.seed(42)
random.shuffle(balanced_data)

# Chia tỉ lệ 70/20/10
total = len(balanced_data)
train = balanced_data[:int(0.7 * total)]
val = balanced_data[int(0.7 * total):int(0.9 * total)]
test = balanced_data[int(0.9 * total):]

# Ghi file JSON
with open("train_t5.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)

with open("val_t5.json", "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

with open("test_t5.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print(f"Dữ liệu đã chia thành: {len(train)} train, {len(val)} val, {len(test)} test")

from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import json

# Load dữ liệu
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        return Dataset.from_list(json.load(f))

train_ds = load_data("train_t5.json")
val_ds = load_data("val_t5.json")

# Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(example):
    input_enc = tokenizer(example["input"], padding="max_length", truncation=True, max_length=512)
    target_enc = tokenizer(example["target"], padding="max_length", truncation=True, max_length=512)

    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_ds = train_ds.map(tokenize)
val_ds = val_ds.map(tokenize)

# Model
model = T5ForConditionalGeneration.from_pretrained("t5-small")

training_args = Seq2SeqTrainingArguments(
    output_dir="./t5_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=10,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer
)

trainer.train()

# Save
model.save_pretrained("./t5_finetuned")
tokenizer.save_pretrained("./t5_finetuned")

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch, json

# Load model đã fine-tuned
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned")
tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned")
model.eval()

with open("test_t5.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch, json, re

# Load mô hình đã fine-tune
model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned")
tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned")
model.eval()

# Load dữ liệu test
with open("test_t5.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Hàm định dạng output cho giống expected
def format_generated(text):
    blocks = re.split(r"- Drugs Name:", text)
    formatted = []

    for block in blocks[1:]:
        lines = block.strip().split(" - ")
        drug_name = lines[0].strip()
        usage = next((l for l in lines if l.startswith("Usage time:")), "Usage time: None")
        dosage = next((l for l in lines if l.startswith("Dosage:")), "Dosage: None")
        formatted.append(f"- Drugs Name: {drug_name}\n- {usage}\n- {dosage}\n")

    return "\n".join(formatted).strip()

# Vòng lặp kiểm tra
results = []
exact_match = 0

for i, sample in enumerate(test_data[:40]):  # 40 mẫu đầu
    prompt = sample['input'].strip()
    expected = sample['target'].strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=512,
            num_beams=4
        )
    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    formatted = format_generated(generated)

    match = formatted.replace(" ", "") == expected.replace(" ", "")
    if match:
        exact_match += 1

    print(f"--- Sample {i+1} ---")
    print(f"Input:\n{prompt}\n")
    print(f"Generated:\n{formatted}\n")
    print(f"Expected:\n{expected}\n")

    results.append({
        "sample": i+1,
        "input": prompt,
        "generated": formatted,
        "expected": expected,
    })



# Lưu file
with open("t5_result.txt", "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"--- Sample {r['sample']} ---\n")
        f.write(f"Input:\n{r['input']}\n\n")
        f.write(f"Generated:\n{r['generated']}\n\n")
        f.write(f"Expected:\n{r['expected']}\n\n")



import json, random
from collections import defaultdict
from datasets import Dataset

with open("input_output.txt", "r", encoding="utf-8") as f:
    content = f.read()

pairs = content.split('---\n')
diag_map = defaultdict(list)

for pair in pairs:
    if not pair.strip():
        continue
    try:
        input_part = pair.split('Input:\n')[1].split('Output:\n')[0].strip()
        output_part = pair.split('Output:\n')[1].strip()
    except IndexError:
        continue

    record = {
        "input": input_part,
        "target": output_part
    }

    diagnose_line = next((line for line in input_part.split("\n") if "Diagnose:" in line), "").lower()
    diagnosis = diagnose_line.replace("diagnose:", "").strip()

    if "cerebral infarction" in diagnosis:
        diag_map["cerebral infarction"].append(record)
    elif "seizure" in diagnosis or "epilepsy" in diagnosis:
        diag_map["seizure"].append(record)
    elif "sleep" in diagnosis:
        diag_map["sleep disorders"].append(record)
    elif "normal" in diagnosis:
        diag_map["normal"].append(record)
    else:
        diag_map["unknown"].append(record)

print("Số lượng mẫu ban đầu:")
for k, v in diag_map.items():
    print(f"  - {k}: {len(v)}")

target_per_class = 1753
balanced_data = []
for key in ["cerebral infarction", "seizure", "sleep disorders", "normal"]:
    records = diag_map[key]
    if not records:
        continue
    times = (target_per_class + len(records) - 1) // len(records)
    extended = (records * times)[:target_per_class]
    balanced_data.extend(extended)

print(f"\nTổng số mẫu sau cân bằng: {len(balanced_data)}")

random.seed(42)
random.shuffle(balanced_data)

train = balanced_data[:int(0.7 * len(balanced_data))]
val = balanced_data[int(0.7 * len(balanced_data)):int(0.9 * len(balanced_data))]
test = balanced_data[int(0.9 * len(balanced_data)):]

with open("train_llama.json", "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open("val_llama.json", "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)
with open("test_llama.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Load mô hình
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=2,
    lora_alpha=4,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(base_model, lora_config)
model = model.to("cuda").train()

#CHUẨN HÓA DỮ LIỆU
from datasets import Dataset

def load_and_format(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return Dataset.from_list([
        {"text": f"{x['input'].strip()}\n<|sep|>\n{x['target'].strip()}"} for x in raw
    ])

train_dataset = load_and_format("train_llama.json")
val_dataset = load_and_format("val_llama.json")
test_dataset = load_and_format("test_llama.json")

def tokenize(example):
    enc = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
    enc["labels"] = enc["input_ids"].copy()
    return enc

train_dataset = train_dataset.map(tokenize)
val_dataset = val_dataset.map(tokenize)

#TRAINING THỦ CÔNG BẰNG ACCELERATE
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import default_data_collator

accelerator = Accelerator()
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=default_data_collator)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=default_data_collator)

model, train_loader, val_loader = accelerator.prepare(model, train_loader, val_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop (3 epochs)
model.train()
for epoch in range(3):  # Tăng số epoch lên 3
    print(f"🔁 Epoch {epoch + 1}/3")
    for step, batch in enumerate(train_loader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}")

output_dir = "./tinyllama_lora_finetuned"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Đã lưu mô hình vào thư mục: {output_dir}")

#KIỂM TRA VÀ LƯU KẾT QUẢ
model.eval()
results = []

raw_test = load_and_format("test_llama.json")

for i in range(40):
    raw_text = raw_test[i]["text"]
    prompt = raw_text.split("<|sep|>")[0].strip() + "\n<|sep|>\n"
    expected = raw_text.split("<|sep|>")[-1].strip()

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("<|sep|>")[-1].strip()

    result = f"--- Sample {i+1} ---\nInput:\n{prompt}\n\nGenerated:\n{generated}\n\nExpected:\n{expected}\n\nMatch: {generated == expected}\n\n"
    print(result)
    results.append(result)

with open("Llama_result.txt", "w", encoding="utf-8") as f:
    f.writelines(results)

print("Đã kiểm tra và lưu kết quả vào Llama-Finetune.txt")
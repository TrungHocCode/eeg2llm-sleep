from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset
import json
import os


def train_model():
    """
    Huấn luyện mô hình GPT-2 với dữ liệu đã chuẩn bị.
    """
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs("./logs", exist_ok=True)
    
    # Load dữ liệu
    train_dataset = load_dataset("train_gpt2.json")
    val_dataset = load_dataset("val_gpt2.json")
    
    # Tokenizer & Model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 không có pad_token mặc định
    
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Tokenization
    train_dataset = train_dataset.map(lambda x: tokenize(x, tokenizer))
    val_dataset = val_dataset.map(lambda x: tokenize(x, tokenizer))
    
    # Thiết lập tham số huấn luyện
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        output_dir="./gpt2_finetuned",
        learning_rate=5e-5,
        weight_decay=0.1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=8,
        logging_dir="./logs",
        logging_steps=20,
        fp16=True,
        lr_scheduler_type="cosine",  # Sử dụng cosine learning rate scheduler
        warmup_ratio=0.1,  # Tăng dần learning rate trong giai đoạn đầu
        metric_for_best_model="loss",
    )
    
    # Khởi tạo Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Huấn luyện mô hình
    print("Bắt đầu huấn luyện mô hình...")
    trainer.train()
    print("Huấn luyện hoàn tất!")
    
    # Lưu mô hình
    print("Lưu mô hình đã huấn luyện...")
    trainer.save_model("./results/fine_tuned_gpt2")
    tokenizer.save_pretrained("./results/fine_tuned_gpt2")
    print("Đã lưu mô hình vào thư mục './results/fine_tuned_gpt2'")


def load_dataset(path):
    """
    Tải tập dữ liệu từ file JSON.
    
    Args:
        path: Đường dẫn đến file JSON
        
    Returns:
        Dataset object
    """
    with open(path, "r", encoding="utf-8") as f:
        return Dataset.from_list(json.load(f))


def tokenize(example, tokenizer):
    """
    Tokenize dữ liệu văn bản.
    
    Args:
        example: Một mẫu dữ liệu
        tokenizer: Tokenizer để xử lý văn bản
        
    Returns:
        Dữ liệu đã được tokenize
    """
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()  # Để Trainer biết cần học từ input_ids
    return tokenized


if __name__ == "__main__":
    # Tạo thư mục results nếu chưa tồn tại
    train_model()
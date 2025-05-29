import re

#Hàm parse từng thuốc từ chuỗi văn bản
def parse_section(text):
    pattern = r"- Drugs Name: (.*?)\n- Usage time: (.*?)\n- Dosage: (.*?)\n"
    return [match for match in re.findall(pattern, text)]

#Hàm tách toàn bộ các mẫu (generated, expected) từ file
def extract_all_samples(file_text):
    samples = file_text.split("--- Sample ")[1:]  # Bỏ phần đầu
    result = []
    for s in samples:
        try:
            # Trích phần Generated
            gen = re.search(r"Generated:\n(.*?)Expected:", s, re.DOTALL).group(1).strip()
            # Trích phần Expected
            exp = re.search(r"Expected:\n(.*?)$", s, re.DOTALL).group(1).strip()

            gen_list = parse_section(gen)
            exp_list = parse_section(exp)

            result.append((gen_list, exp_list))
        except:
            continue
    return result

#Hàm tính Accuracy theo công thức tùy chỉnh của bạn
def custom_accuracy(samples):
    total_score = 0
    max_possible_score = 0

    for generated, expected in samples:
        expected_matched = [False] * len(expected)
        generated_matched = [False] * len(generated)

        # So khớp từng thuốc trong expected
        for i, exp in enumerate(expected):
            best_score = 0
            best_j = -1
            for j, gen in enumerate(generated):
                if generated_matched[j]:
                    continue
                score = 0
                if gen[0] == exp[0]:  # Đúng thuốc
                    score += 6
                    if gen[1] == exp[1]:  # Đúng thời gian
                        score += 1
                    if gen[2] == exp[2]:  # Đúng liều
                        score += 1
                if score > best_score:
                    best_score = score
                    best_j = j
            if best_score > 0:
                expected_matched[i] = True
                generated_matched[best_j] = True
                total_score += best_score

            max_possible_score += 7  # mỗi thuốc expected tối đa 7 điểm


    accuracy = total_score / max_possible_score if max_possible_score > 0 else 0
    return accuracy

#Đọc file
with open("result_after_finetune.txt", "r", encoding="utf-8") as f:
    text_gpt2 = f.read()
with open("t5_result.txt", "r", encoding="utf-8") as f:
    text_t5 = f.read()
with open("Llama_result.txt", "r", encoding="utf-8") as f:
    text_t5 = f.read()

#Tính accuracy
samples_gpt2 = extract_all_samples(text_gpt2)
accuracy_gpt2 = custom_accuracy(samples_gpt2)

samples_t5 = extract_all_samples(text_t5)
accuracy_t5 = custom_accuracy(samples_t5)

samples_llama = extract_all_samples(text_t5)
accuracy_llama = custom_accuracy(samples_llama)

print(f"Accuracy GPT2: {accuracy_gpt2*100:.2f}%")
print(f"Accuracy T5: {accuracy_t5*100:.2f}%")
print(f"Accuracy Llama: {accuracy_llama*100:.2f}%")


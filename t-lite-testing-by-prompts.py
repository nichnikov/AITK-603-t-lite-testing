import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)

model_name = "t-tech/T-pro-it-1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype="auto",
    device_map="auto",
)

tsxs_files_names = ["Поздравление_1.txt", 
                    "Поздравление_2.txt",
                    "Поздравление_3.txt",
                    "Поздравление_4.txt",
                    "Поздравление_5.txt",
                    "prompt1_арбитражка.txt"]

for fn in tsxs_files_names:
    with open(os.path.join("data", "prompt1_арбитражка.txt"), "r") as f:
        prompt = f.read()


# prompt = "Напиши стих про машинное обучение"
messages = [
    {"role": "system", "content": "Ты T-pro, виртуальный ассистент в Т-Технологии. Твоя задача - быть полезным диалоговым ассистентом."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=256
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


print('\n\n')
print(response)
print('\n\n')
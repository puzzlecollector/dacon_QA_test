import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType

model_name = "beomi/open-llama-2-ko-7b"
model_dir = "open-llama-2-ko-7b"
version = 1 
epochs_ = 10
device = torch.device("cuda")

# Load training data
train = pd.read_csv("train.csv")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, eos_token="</s>", padding_side="left")

# Prepare formatted data
formatted_data = []
for _, row in tqdm(train.iterrows(), desc="Formatting Data"):
    for q_col in ["질문_1", "질문_2"]:
        for a_col in ["답변_1", "답변_2", "답변_3", "답변_4", "답변_5"]:
            input_text = row[q_col] + tokenizer.eos_token + row[a_col]
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            formatted_data.append(input_ids.to(torch.device("cuda")))

# Load model and apply LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    # Specify target modules more precisely
    target_modules=["q_proj", "k_proj", "v_proj"]  # Adjusted to target query, key, value projection layers
)

# Wrap the base model with LoRA
model = get_peft_model(base_model, lora_config)
model.to(torch.device("cuda"))

# Training configuration
CFG = {"LR": 2e-5, "EPOCHS": epochs_}
optimizer = AdamW(model.parameters(), lr=CFG['LR'])

# Fine-tuning loop
model.train()
for epoch in range(CFG['EPOCHS']):
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm(formatted_data, desc=f"Epoch {epoch+1}")):
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{CFG['EPOCHS']}, Average Loss: {total_loss / len(formatted_data)}")

# Save the fine-tuned model
model_save_path = f"./hansol-{model_dir}_v{version}.pth"
torch.save(model.state_dict(), model_save_path)

# Correctly loading the fine-tuned model with LoRA for inference
# Initialize the base model and LoRA configuration as done before fine-tuning
test = pd.read_csv("test.csv") 

model_save_path = f"./hansol-{model_dir}_v{version}.pth"

base_model_for_inference = AutoModelForCausalLM.from_pretrained(model_name)
lora_config_for_inference = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

# Wrap the base model with the same LoRA configuration used for fine-tuning
model_for_inference = get_peft_model(base_model_for_inference, lora_config_for_inference)

# Load the fine-tuned model's state dictionary
model_for_inference.load_state_dict(torch.load(model_save_path))

# Ensure to move the model to the correct device
model_for_inference.to(torch.device("cuda"))

# test.csv의 '질문'에 대한 '답변'을 저장할 리스트
preds = []

# '질문' 컬럼의 각 질문에 대해 답변 생성
for test_question in tqdm(test['질문'], position=0, leave=True, desc="Inference"):
    # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
    input_ids = tokenizer.encode(test_question + tokenizer.eos_token, return_tensors='pt')

    # 답변 생성
    output_sequences = model_for_inference.generate(
        input_ids=input_ids.to(device),
        max_length=300,
        temperature=0.9,
        top_k=1,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1
    )

    # 생성된 텍스트(답변) 저장
    for generated_sequence in output_sequences:
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
        # 질문과 답변의 사이를 나타내는 eos_token (</s>)를 찾아, 이후부터 출력
        answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
        answer_only = full_text[answer_start:].strip()
        answer_only = answer_only.replace('\n', ' ')
        preds.append(answer_only)
        

preds_df = pd.DataFrame(preds, columns=['Generated Answer'])
# Define the CSV file path
output_csv_path = "generated_preds.csv"
# Save the DataFrame to a CSV file
preds_df.to_csv(output_csv_path, index=False, encoding='utf-8')
print(f"Predictions saved to {output_csv_path}")


# Test 데이터셋의 모든 질의에 대한 답변으로부터 512 차원의 Embedding Vector 추출
# 평가를 위한 Embedding Vector 추출에 활용하는 모델은 'distiluse-base-multilingual-cased-v1' 이므로 반드시 확인해주세요.
from sentence_transformers import SentenceTransformer # SentenceTransformer Version 2.2.2

# Embedding Vector 추출에 활용할 모델(distiluse-base-multilingual-cased-v1) 불러오기
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
pred_embeddings = model.encode(preds)
print(pred_embeddings.shape) 

submit = pd.read_csv('./sample_submission.csv')
# 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
submit.iloc[:,1:] = pred_embeddings
submit.head()
# 리더보드 제출을 위한 csv파일 생성
submit.to_csv(f'{model_dir}_submission_v{version}.csv', index=False)


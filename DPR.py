import pandas as pd
import numpy as np 
import torch
from transformers import *
from tqdm import tqdm 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler 
from tqdm.auto import tqdm 
import faiss 

train = pd.read_csv("train.csv")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

questions, answers = [], [] 

for _, row in tqdm(train.iterrows()):
    for q_col in ["질문_1", "질문_2"]:
        for a_col in ["답변_1", "답변_2", "답변_3", "답변_4", "답변_5"]:
            questions.append(row[q_col])
            answers.append(row[a_col])
            
            
q_input_ids, q_attn_masks = [], []
a_input_ids, a_attn_masks = [], [] 

for i in tqdm(range(len(questions)), position=0, leave=True):
    encoded_question = tokenizer(questions[i], max_length=512, truncation=True, padding="max_length") 
    q_input_ids.append(encoded_question["input_ids"]) 
    q_attn_masks.append(encoded_question["attention_mask"]) 
    
    encoded_answer = tokenizer(answers[i], max_length=512, truncation=True, padding="max_length") 
    a_input_ids.append(encoded_answer["input_ids"]) 
    a_attn_masks.append(encoded_answer["attention_mask"])
    
q_input_ids = torch.tensor(q_input_ids, dtype=int) 
q_attn_masks = torch.tensor(q_attn_masks, dtype=int) 

a_input_ids = torch.tensor(a_input_ids, dtype=int) 
a_attn_masks = torch.tensor(a_attn_masks, dtype=int) 

batch_size = 32
train_data = TensorDataset(q_input_ids, q_attn_masks, a_input_ids, a_attn_masks) 
train_sampler = SequentialSampler(train_data) 
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size) 

model_name = "monologg/kobigbird-bert-base"
question_embedder = AutoModel.from_pretrained(model_name) 
answer_embedder = AutoModel.from_pretrained(model_name) 
device = torch.device('cuda') 

question_embedder.to(device) 
answer_embedder.to(device) 

epochs = 30
params = list(question_embedder.parameters()) + list(answer_embedder.parameters()) 
optimizer = torch.optim.AdamW(params, lr=2e-5) 
t_total = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*t_total), num_training_steps=t_total)

question_embedder.zero_grad() 
answer_embedder.zero_grad() 
torch.cuda.empty_cache() 
for epoch in tqdm(range(epochs), position=0, leave=True, total=epochs):
    train_loss = 0 
    question_embedder.train() 
    answer_embedder.train() 
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="training"):
        batch = tuple(t.to(device) for t in batch)
        question_input_ids, question_attn_masks, answer_input_ids, answer_attn_masks = batch 
        question_embeddings = question_embedder(question_input_ids, question_attn_masks).pooler_output
        answer_embeddings = answer_embedder(answer_input_ids, answer_attn_masks).pooler_output 
        sim_scores = torch.matmul(question_embeddings, torch.transpose(answer_embeddings, 0, 1)) 
        targets = torch.arange(0, question_embeddings.shape[0]).long().to(device) 
        sim_scores = F.log_softmax(sim_scores, dim=1) 
        loss = F.nll_loss(sim_scores, targets) 
        train_loss += loss.item() 
        loss.backward() 
        optimizer.step() 
        scheduler.step() 
        optimizer.zero_grad() 
    
    avg_train_loss = train_loss / len(train_dataloader)  
    print(f"Epoch:{epoch+1} | avg train loss:{avg_train_loss}")
    
torch.save(question_embedder.state_dict(), "question_embedder_DPR.pt") 
torch.save(answer_embedder.state_dict(), "answer_embedder_DPR.pt")
print("done saving!") 

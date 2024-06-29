from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

input_ids = encoded_inputs['input_ids']
attention_mask = encoded_inputs['attention_mask']
dataset = TensorDataset(input_ids, attention_mask)

if not os.path.exists('models/roberta'):
    raise Exception("Model not found. Please download the model first. Link: artoriusss/roberta-text-classify/pyTorch/roberta")

batch_size = 16  # Adjust batch size based on your GPU memory
dataloader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)

model.eval()

all_predictions = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):  # Add tqdm to the loop
        batch_input_ids, batch_attention_mask = batch
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        logits = outputs.logits
        
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())

test_preprocessed_df['predicted_label'] = all_predictions
test_preprocessed_df.to_csv('test_predictions.csv', index=False)
import pandas as pd
import torch
from tqdm import tqdm
from src.config import DATA_DIR, DEVICE
from src.model import model, tokenizer
import os

def evaluate_and_save():
    model.eval()
    dataset_test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    preds = []

    for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
        inputs = tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
            preds.extend(pred)

    dataset_test['target'] = preds
    dataset_test.to_csv(os.path.join('output', 'output.csv'), index=False)

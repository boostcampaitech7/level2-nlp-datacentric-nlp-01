import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import DEVICE, MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=7).to(DEVICE)

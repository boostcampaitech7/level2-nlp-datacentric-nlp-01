import LabelCorrector
from abc import ABC, abstractmethod
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

from src.config import OUTPUT_DIR

class LabelCorrectorbyModel(LabelCorrector):
        
        @abstractmethod
        def correct(self, df: pd.DataFrame) -> pd.DataFrame:
            """dataframe의 모든 행에 noise가 없다고 가정하고, 
            모든 행의 label을 교정한다.
    
            Args:
                df (pd.DataFrame): noise가 없는 dataframe
    
            Returns:
                pd.DataFrame: label이 교정된 dataframe
            """
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(OUTPUT_DIR, 'model'))
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(OUTPUT_DIR, 'model'))
            model.eval()
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer.to("cuda" if torch.cuda.is_available() else "cpu")
            for i in range(len(df)):
                text = df['text'][i]
                tokenized_input = tokenizer(text, padding='max_length', truncation=True, return_tensors='pt').to("cuda" if torch.cuda.is_available() else "cpu")
                with torch.no_grad():
                    output = model(**tokenized_input)
                label = torch.argmax(output.logits).item()
                df['target'][i] = label

            return df
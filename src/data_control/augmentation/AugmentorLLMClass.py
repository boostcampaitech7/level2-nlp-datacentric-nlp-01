import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import json
from tqdm import tqdm
from .Augmentor import Augmentor

class AugmentorLLM(Augmentor):
    
    def __init__(self, model_id: str = "rtzr/ko-gemma-2-9b-it", prompt: str = 'prompts/data-aug.json', code='LLM'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        self.code = code
        with open(prompt, "r", encoding="utf-8") as file:
            self.prompt = json.load(file)
        
    def _augment_row(self, row: pd.Series) -> str:
        """행 하나에 대해서 데이터 증강하는 함수

        Args:
            row (pd.Series): 증강할 행

        Returns:
            str: 데이터 증강 결과
        """
        
        text = row['text']
        
        messages = self.prompt
        messages.append({"role": "user", "content": f"text: '{text}'"})
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=100,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )
        
        output = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        aug_text = re.search(r"text:\s*'(.+)'", output)
        aug_text = aug_text.group(1).strip() if aug_text else ""

        return aug_text
    
    def augment(self, df: pd.DataFrame, only_aug: bool = False) -> pd.DataFrame:
        """데이터프레임 전체에 대해 데이터 증강하는 함수

        Args:
            df (pd.DataFrame): 증강할 데이터프레임

        Returns:
            pd.DataFrame: 데이터 증강 결과
        """
        results = [] 
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            aug_text = self._augment_row(row)
            row['ID'] = row['ID'] + '_' + self.code
            row['aug_text'] = aug_text
            results.append(row)
        
        results_df = pd.DataFrame(results)
        if not only_aug:
            results_df = pd.concat([df, results_df], ignore_index=True)
        return results_df
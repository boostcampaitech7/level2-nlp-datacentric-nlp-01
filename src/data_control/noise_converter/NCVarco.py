import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm
from .NoiseConverter import NoiseConverter
from typing import Tuple, Optional
import json

class NCVarco(NoiseConverter):
    
    def __init__(self, model_id: str = "NCSOFT/Llama-VARCO-8B-Instruct", prompt: str = 'prompts/varco-converter.json'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        with open(prompt, "r", encoding="utf-8") as file:
            self.prompt = json.load(file)
        
    def _convert_row(self, row: pd.Series) -> Tuple[str, int, int, Optional[str]]:
        """행 하나에 대해서 노이즈를 복원하는 함수

        Args:
            row (pd.Series): 노이즈가 있는 행

        Returns:
            Tuple[str, int, int, Optional[str]]: 
            - 복원된 텍스트
            - 텍스트의 label (0~6)
            - 노이즈가 있는지 여부 (1: 노이즈 있음, 0: 노이즈 없음)
            - reason : 텍스트 복원 방법에 대한 설명 (Optional)
        """
        
        text = row['text']
        label = row['target']
        
        messages = self.prompt + [{"role": "user", "content": f"text: '{text}'\nlabel: {label}"}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        ).to(self.model.device)

        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            eos_token_id=eos_token_id,
            max_length=8192
        )
        
        output = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        
        cleaned_text = re.search(r"text:\s'([^']*)'", output).group(1) if re.search(r"text:\s'([^']*)'", output) else ""
        label = int(re.search(r"label:\s(\d+)", output).group(1)) if re.search(r"label:\s(\d+)", output) else ""
        noise = re.search(r"noise:\s([10])", output).group(1) if re.search(r"noise:\s([10])", output) else ""
        reason_match = re.search(r"reason:\s(.+)", output)
        reason = reason_match.group(1).strip() if reason_match else ""

        return cleaned_text, label, noise, reason
    
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """노이즈가 존재하는 dataframe을 받아 입력한 프롬프트로 노이즈를 복원하여 반환하는 function

        Args:
            df (pd.DataFrame): 모든 행에 noise가 존재하는 dataframe

        Returns:
            pd.DataFrame: noise가 복원된 dataframe
        """
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            cleaned_text, label, noise, reason = self._convert_row(row)
            results.append({
                'ID': row['ID'],
                'original': row['text'],
                'text': cleaned_text,
                'target': label,
                'noise': noise,
                # 'reason': reason
            })
        
        results_df = pd.DataFrame(results)
        return results_df
    
def routine_nc_varco():
    df = pd.read_csv('data/train.csv')
    converter = NCVarco()
    converted_df = converter.convert(df)
    converted_df.to_csv('data/converted_varco.csv', index=False)
    
if __name__ == '__main__':
    routine_nc_varco()
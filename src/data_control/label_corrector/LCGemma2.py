import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from kiwipiepy import Kiwi
from tqdm import tqdm
from .LabelCorrector import LabelCorrector
from typing import List
from collections import Counter

class LCGemma2(LabelCorrector):
    
    def __init__(self, model_id: str = "rtzr/ko-gemma-2-9b-it", morph_file: str = 'data/train.csv'):
        """Gemma2 모델을 이용한 레이블 교정 클래스

        Args:
            model_id (str, optional): 사용할 모델의 이름. Defaults to "rtzr/ko-gemma-2-9b-it".
            morph_file (str, optional): 형태소 분석을 위해 넘겨줄 csv 파일명. Defaults to 'data/train.csv'.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        
        self.top_nouns = self._count_morphs(pd.read_csv(morph_file))
        
    def _count_morphs(self, df: pd.DataFrame, common: int = 30) -> pd.DataFrame:
        def extract_nouns(text):
            kiwi = Kiwi()
            nouns = []
            result = kiwi.analyze(text)
            for token, pos, _, _ in result[0][0]:
                if len(token) != 1 and (pos.startswith('N') or pos.startswith('SL')):
                    nouns.append(token)
            return nouns
        
        top_nouns = {}
        for target_value in range(7):
            # 특정 target에 속하는 텍스트 모음
            texts = df[df['target'] == target_value]['text']
            nouns = []

            # 명사 추출 및 빈도수 계산
            for text in texts:
                nouns.extend(extract_nouns(text))
            
            # 상위 50개의 명사 추출 -> 30개
            noun_counts = Counter(nouns)
            top_nouns[f'target_{target_value}'] = [word for word, _ in noun_counts.most_common(common)]
            
        return pd.DataFrame(top_nouns)
    
    def _correct_row(self, row: pd.Series):
        text = row['text']
        
        messages = [
            {
                "role": "system",
                "content": (f"""
                            너는 뛰어난 분류기로써 0부터 6까지 해당하는 분류에 대해 교정해주는 역할을 아주 섬세하게 잘 해
                            각 주제가 모호해 보일지라도 가장 그럴 듯한 하나의 분류를 제시하지
                            그리고 각 분류의 핵심 단어는 이하와 같아
                            target 0: {list(self.top_nouns['target_0'])}
                            target 1: {list(self.top_nouns['target_1'])}
                            target 2: {list(self.top_nouns['target_2'])}
                            target 3: {list(self.top_nouns['target_3'])}
                            target 4: {list(self.top_nouns['target_4'])}
                            target 5: {list(self.top_nouns['target_5'])}
                            target 6: {list(self.top_nouns['target_6'])}
                            이제 너에게 text를 건넬테니 맞는 주제를 분류해내봐
                            반드시 너는 target: 이 뒤를 이어서 대답해야해 숫자 단 하나로
                            명심해 하나의 숫자(0,1,2,3,4,5,6) 중 하나로 대답해야해 핵심 단어들을 잘 고려해봐
                            이제 바로 text를 건네줄게""")},
                {"role": "user", "content": f"text: {text}"}]
        
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
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.8,
            top_p=1.0,
        )
        
        output = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return_target = re.search(r"target:\s*(\d+)", output)
        return_target = return_target.group(1).strip() if return_target else ""
        
        return return_target
    
    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            target = self._correct_row(row)
            row['trimed_target'] = target
            row['ID'] = row['ID'] + '_with_LLM_Trim_target'
            results.append(row)
            
        corrected_df = pd.DataFrame(results)
        
        return corrected_df

def routine_gemma2():
    df = pd.read_csv('data/corrected_BLEU.csv')
    corrector = LCGemma2()
    corrected_df = corrector.correct(df)
    corrected_df.to_csv('data/BLEU_with_LLM_Trim_target.csv', index=False)
    
    return corrected_df

if __name__ == "__main__":
    routine_gemma2()
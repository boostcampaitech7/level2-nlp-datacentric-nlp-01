import pandas as pd
from koeda import SR, RI, RD, RS, AEDA
from typing import List, Union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from functools import partial
from Augmentor import Augmentor

class EasyDataAugmentation(Augmentor):
    def __init__(
        self,
        methods: List[str] = ['SR', 'RI', 'RD', 'RS', 'AEDA'],
        morpheme_analyzer: str = "Mecab",
        p: float = 0.3,
        repetition: int = 1
    ):
        """
        Args:
            methods (List[str]): 사용할 augmentation 방법들의 리스트
            morpheme_analyzer (str): 형태소 분석기 선택 ("Mecab" 등)
            p (float): augmentation 적용 확률
            repetition (int): 각 텍스트당 생성할 augmentation 데이터 수
        """
        self.methods = methods
        self.morpheme_analyzer = morpheme_analyzer
        self.p = p
        self.repetition = repetition
        
        # augmenter 초기화
        self.augmenters = {}
        for method in methods:
            if method == 'SR':
                self.augmenters['SR'] = SR(morpheme_analyzer=morpheme_analyzer)
            elif method == 'RI':
                self.augmenters['RI'] = RI(morpheme_analyzer=morpheme_analyzer)
            elif method == 'RD':
                self.augmenters['RD'] = RD(morpheme_analyzer=morpheme_analyzer)
            elif method == 'RS':
                self.augmenters['RS'] = RS(morpheme_analyzer=morpheme_analyzer)
            elif method == 'AEDA':
                self.augmenters['AEDA'] = AEDA(morpheme_analyzer=morpheme_analyzer)
    
    def augment_text(self, text: str) -> List[str]:
        """하나의 텍스트에 대해 선택된 모든 augmentation 방법을 적용

        Args:
            text (str): 원본 텍스트

        Returns:
            List[str]: augmentation이 적용된 텍스트들의 리스트
        """
        augmented_texts = []
        
        for method, augmenter in self.augmenters.items():
            try:
                result = augmenter(
                    text,
                    p=self.p,
                    repetition=self.repetition
                )
                # 결과가 리스트가 아닌 경우 리스트로 변환
                if isinstance(result, str):
                    result = [result]
                augmented_texts.extend(result)
            except Exception as e:
                print(f"Warning: {method} augmentation failed for text '{text}': {str(e)}")
                continue
                
        return augmented_texts

    def augment(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """DataFrame의 모든 행에 augmentation을 적용

        Args:
            df (pd.DataFrame): augmentation을 적용할 DataFrame
            text_column (str): 텍스트가 포함된 열 이름

        Returns:
            pd.DataFrame: augmentation이 적용된 DataFrame
        """
        # 원본 DataFrame을 복사
        augmented_df = df.copy()
        
        # 각 행에 대해 augmentation 적용
        all_augmented_rows = []
        
        for idx, row in df.iterrows():
            # 원본 텍스트에 대해 augmentation 적용
            augmented_texts = self.augment_text(row[text_column])
            
            # 각 augmented 텍스트에 대해 새로운 행 생성
            for aug_text in augmented_texts:
                new_row = row.copy()
                new_row[text_column] = aug_text
                all_augmented_rows.append(new_row)
        
        # 모든 augmented 행들을 DataFrame으로 변환
        if all_augmented_rows:
            augmented_data = pd.DataFrame(all_augmented_rows)
            
            # 원본 데이터와 augmented 데이터를 합침
            result_df = pd.concat([augmented_df, augmented_data], ignore_index=True)
            # DataFrame의 행을 섞음
            result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
            return result_df
        
        return augmented_df
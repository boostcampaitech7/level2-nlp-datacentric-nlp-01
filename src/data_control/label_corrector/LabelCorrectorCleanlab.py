from src.data_control.label_corrector.LabelCorrector import LabelCorrector
from config import SEED, DEVICE
import torch

from cleanlab.filter import find_label_issues

import pandas as pd
import numpy as np

from typing import List, Tuple

class LabelCorrectorCleanlab(LabelCorrector):
        """
        Cleanlab를 이용한 라벨 교정 클래스
        
        Attributes:
            model: 학습된 모델
            seed: random seed
        """
        
        def __init__(self, model):
            self.model = model
            self.seed = SEED

        def find_issues(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
            """
            Cleanlab을 이용해 라벨 이슈를 찾는 메소드
            
            Args:
                df: 라벨 이슈를 찾을 데이터프레임
                
            Returns:
                label_issues: 라벨 이슈가 있는 인덱스 리스트
            """
            
            self.model.eval().to(DEVICE)
            
            # 모델을 이용해 라벨 예측하고 확률 계산
            with torch.no_grad():
                data = self.model.tokenizer(df['text'].tolist(), padding='max_length', truncation=True, return_tensors='pt')
                data = {k: v.to(DEVICE) for k, v in data.items()}
                outputs = self.model(**data)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
            
            labels = np.array(df['target'].tolist())

            # Cleanlab을 이용해 라벨 이슈를 찾음
            label_issues = find_label_issues(
                labels=labels,
                pred_probs=probs,
                return_indices_ranked_by='self_confidence'  # 신뢰도가 낮은 순으로 정렬된 인덱스 반환
            )

            print(f"Cleanlab이 {len(label_issues)}개의 라벨 이슈를 발견했습니다.")

            return probs, label_issues
        
        def correct(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Cleanlab을 이용해 라벨을 교정하는 메소드

            Args:
                df: 라벨을 교정할 데이터프레임

            Returns:
                df: 라벨이 교정된 데이터프레임
            """
            
            probs, label_issues = self.find_issues(df)

            # 라벨 이슈가 있는 인덱스에 대해 교정 작업 수행
            for idx in label_issues:
                df.loc[idx, 'target'] = np.argmax(probs[idx])
                
                print(f"인덱스 {idx}: 라벨 교정됨")

            return df  #수정된 데이터프레임 반환
        
        def clean(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Cleanlab을 이용해 이상 라벨들을 제거하는 메소드

            Args:
                df: 라벨을 교정할 데이터프레임

            Returns:
                df: 라벨이 교정된 데이터프레임
            """
            
            _, label_issues = self.find_issues(df)

            # 라벨 이슈가 있는 인덱스에 대해 제거 작업 수행
            for idx in label_issues:
                df.drop(idx, inplace=True)
                
                print(f"인덱스 {idx}: 라벨 이슈 제거됨")

            return df
from src.data_control.label_corrector.LabelCorrector import LabelCorrector
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
        
        def __init__(self, model, tokenizer, device):
            """
            Cleanlab을 이용한 라벨 교정 클래스 초기화

            Args:
                model: 학습된 모델
                tokenizer: 토크나이저
                device: 사용할 디바이스
            """
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def find_issues(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
            """
            Cleanlab을 이용해 라벨 이슈를 찾는 메소드
            
            Args:
                df: 라벨 이슈를 찾을 데이터프레임
                
            Returns:
                label_issues: 라벨 이슈가 있는 인덱스 리스트
            """
            
            self.model.eval().to(self.device)
            
            # 모델을 이용해 라벨 예측하고 확률 계산
            batch_size = 32  # 배치 크기 설정
            probs_list = []
            labels_list = []

            with torch.no_grad():
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i + batch_size]
                    data = self.tokenizer(batch_df['text'].tolist(), padding='max_length', truncation=True, return_tensors='pt')
                    data = {k: v.to(self.device) for k, v in data.items()}
                    outputs = self.model(**data)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
                    probs_list.append(probs)
                    labels_list.extend(batch_df['target'].tolist())

            probs = np.concatenate(probs_list, axis=0)
            labels = np.array(labels_list)

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
            df = df.drop(label_issues)
            print(f"Cleanlab을 이용해 {len(label_issues)}개의 이상 라벨을 제거했습니다.")

            return df  #수정된 데이터프레임 반환
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoTokenizer
from typing import List
from sentence_transformers import SentenceTransformer, util
import os

class AugmentorTester:
    """
    데이터 증강 결과의 품질을 테스트하는 클래스.
    STS와 BLEU 스코어를 사용하여 원본 텍스트와 증강된 텍스트 간의 유사도를 측정.
    """
    
    def __init__(self):
        """
        AugmentorTester 초기화.
        문장 임베딩 모델과 한국어 BERT 토크나이저를 로드.
        """
        self.semantic_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        self.tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
        
    def test_augmentation_quality(self, combined_df: pd.DataFrame) -> dict:
        """
        데이터 증강 결과 테스트.

        Args:
            combined_df (pd.DataFrame): 원본 텍스트와 증강된 텍스트가 포함된 DataFrame.
                                      'text'와 'ID' 컬럼이 필수.

        Returns:
            dict: 테스트 결과를 포함하는 딕셔너리.
                 semantic_similarity: STS 점수
                 text_similarity: BLEU 점수
                 all_results: 모든 텍스트 쌍에 대한 결과
        """
        # 데이터 검증
        if 'text' not in combined_df.columns or 'ID' not in combined_df.columns:
            raise ValueError("DataFrame must contain 'text' and 'ID' columns")
        
        if len(combined_df) % 2 != 0:
            raise ValueError("DataFrame must contain even number of rows")
        
        # 컬럼명 행을 제외한 실제 데이터 수
        data_len = len(combined_df)
        half_len = data_len // 2  # 정확히 절반
        print(f"half_len: {half_len}")# 데이터 분리 디버깅
        # 데이터 분리
        original_df = combined_df.iloc[0:half_len].copy()  # 첫 절반
        augmented_df = combined_df.iloc[half_len:].copy()  # 나머지 절반
        
        print(f"전체 데이터 수(컬럼명 제외): {data_len}")
        print(f"원본 데이터: {len(original_df)}")
        print(f"증강 데이터: {len(augmented_df)}")
        
        # 각 행별 결과 저장
        all_results = []
        for (i, orig_row), (j, aug_row) in zip(original_df.iterrows(), augmented_df.iterrows()):
            if pd.isna(orig_row.text) or pd.isna(aug_row.text):
                continue
                

            emb1 = self.semantic_model.encode(orig_row.text, convert_to_tensor=True)
            emb2 = self.semantic_model.encode(aug_row.text, convert_to_tensor=True)
            
            
            sts_score = float(util.pytorch_cos_sim(emb1, emb2)[0][0])
            print(f"\n원본: {orig_row.text}")
            print(f"\n증강: {aug_row.text}")            
            print(f"코사인 유사도: {sts_score:.4f}")
            
            bleu_score = sentence_bleu(
                [self.tokenize(orig_row.text)],
                self.tokenize(aug_row.text)
            )
            
            all_results.append({
                'original_text': orig_row.text,
                'augmented_text': aug_row.text,
                'sts_score': sts_score,
                'bleu_score': bleu_score,
                'original_id': orig_row.ID,
                'augmented_id': aug_row.ID
            })
        
        sts_scores = [r['sts_score'] for r in all_results]
        bleu_scores = [r['bleu_score'] for r in all_results]
        
        results = {
            'semantic_similarity': {
                'mean_sts': np.mean(sts_scores),
                'min_sts': np.min(sts_scores),
                'max_sts': np.max(sts_scores),
            },
            'text_similarity': {
                'mean_bleu': np.mean(bleu_scores),
                'min_bleu': np.min(bleu_scores),
                'max_bleu': np.max(bleu_scores)
            },
            'all_results': all_results  # 모든 행의 결과 저장
        }
        
        return results
    
    def tokenize(self, x: str) -> List[str]:
        """
        BERT 토크나이저를 사용하여 텍스트를 토큰화합니다.

        Args:
            x (str): 토큰화할 텍스트

        Returns:
            List[str]: 토큰화된 텍스트 리스트
        """
        return self.tokenizer.tokenize(x)
    
    def print_test_results(self, results: dict):
        """
        테스트 결과를 콘솔에 출력합니다.

        Args:
            results (dict): test_augmentation_quality()가 반환한 결과 딕셔너리
        """
        print("\n=== 증강 데이터 품질 테스트 결과 ===")
        
        print("\n1. 의미적 유사도 (STS):")
        print(f"- 평균: {results['semantic_similarity']['mean_sts']:.4f}")
        print(f"- 최소: {results['semantic_similarity']['min_sts']:.4f}")
        print(f"- 최대: {results['semantic_similarity']['max_sts']:.4f}")
        
        print("\n2. 표면적 유사도 (BLEU):")
        print(f"- 평균: {results['text_similarity']['mean_bleu']:.4f}")
        print(f"- 최소: {results['text_similarity']['min_bleu']:.4f}")
        print(f"- 최대: {results['text_similarity']['max_bleu']:.4f}")
        

    
    def save_results(self, results: dict, output_path: str):
        """
        테스트 결과를 CSV 파일로 저장합니다.

        Args:
            results (dict): test_augmentation_quality()가 반환한 결과 딕셔너리
            output_path (str): 결과를 저장할 디렉토리 경로
        
        저장되는 파일:
            - all_results.csv: 모든 텍스트 쌍의 상세 결과
            - augmentation_stats.csv: 전체 통계 정보
        """
        # 모든 행의 결과를 DataFrame으로 변환
        df = pd.DataFrame(results['all_results'])
        
        # 전체 통계 추가
        stats = pd.DataFrame({
            'metric': ['STS_mean', 'STS_min', 'STS_max', 'BLEU_mean', 'BLEU_min', 'BLEU_max'],
            'value': [
                results['semantic_similarity']['mean_sts'],
                results['semantic_similarity']['min_sts'],
                results['semantic_similarity']['max_sts'],
                results['text_similarity']['mean_bleu'],
                results['text_similarity']['min_bleu'],
                results['text_similarity']['max_bleu']
            ]
        })
        
        # 저장
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(f"{output_path}/all_results.csv", index=False)  # 모든 행의 결과
        stats.to_csv(f"{output_path}/augmentation_stats.csv", index=False)  # 전체 통계





# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    combined_df = pd.read_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/train_augmented_deepl_en_1loop.csv')
    
    # 결과 저장 디렉토리 경로 (파일 확장자 제외)
    out_dir = "/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/deepl_outputs_augmentation_test"
    
    # 테스트 실행
    tester = AugmentorTester()
    results = tester.test_augmentation_quality(combined_df)
    
    # 결과 출력 및 저장
    tester.print_test_results(results)
    tester.save_results(results, out_dir)


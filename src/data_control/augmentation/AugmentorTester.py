
import unittest
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

class AugmentorTester:
    def __init__(self):
        self.rouge = Rouge()
        self.semantic_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')
        
    def test_augmentation_quality(self, combined_df: pd.DataFrame) -> dict:
        """
        데이터 품질 테스트
        Args:
            df (pd.DataFrame): 품질 테스트를 진행할 dataframe
        Returns:
            dict: 품질 테스트 결과
        """
        # 데이터 분리
        total_len = len(combined_df)
        mid_point = total_len // 2
        
        original_df = combined_df.iloc[:mid_point].reset_index(drop=True)
        augmented_df = combined_df.iloc[mid_point:].reset_index(drop=True)
        
        results = {
            'basic_stats': self._test_basic_statistics(original_df, augmented_df),
            'semantic_similarity': self._test_semantic_similarity(original_df, augmented_df),
            'text_similarity': self._test_text_similarity(original_df, augmented_df),
        }
        
        return results
    
    def _test_basic_statistics(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> dict:
        """
        데이터 길이 검사
        Args:
            original_df (pd.DataFrame): 원본 데이터
            augmented_df (pd.DataFrame): 증강 데이터
        Returns:
            dict: 데이터 길이 검사 결과
        """
        orig_lengths = original_df['text'].str.len()
        aug_lengths = augmented_df['text'].str.len()
        
        return {
            'original_size': len(original_df),
            'augmented_size': len(augmented_df),
            'avg_length_original': orig_lengths.mean(),
            'avg_length_augmented': aug_lengths.mean(),
            'length_difference': abs(orig_lengths.mean() - aug_lengths.mean())
        }
    
    def _test_semantic_similarity(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> dict:
        """
        의미적 유사도 검사
        Args:
            original_df (pd.DataFrame): 원본 데이터
            augmented_df (pd.DataFrame): 증강 데이터
        Returns:
            dict: 의미적 유사도 검사 결과
        """
        similarities = []
        
        for orig, aug in zip(original_df['text'].head(100), augmented_df['text'].head(100)):
            emb1 = self.semantic_model.encode(orig)
            emb2 = self.semantic_model.encode(aug)
            similarity = 1 - cosine(emb1, emb2)
            similarities.append(similarity)
            
        return {
            'mean_semantic_similarity': np.mean(similarities),
            'min_semantic_similarity': np.min(similarities),
            'max_semantic_similarity': np.max(similarities)
        }
    
    def _test_text_similarity(self, original_df: pd.DataFrame, augmented_df: pd.DataFrame) -> dict:
        """
        BLEU와 ROUGE 점수를 통한 텍스트 유사도 검사
        Args:
            original_df (pd.DataFrame): 원본 데이터
            augmented_df (pd.DataFrame): 증강 데이터
        Returns:
            dict: 텍스트 유사도 검사 결과
        """
        bleu_scores = []
        rouge_scores = []
        
        for orig, aug in zip(original_df['text'], augmented_df['text']):
            # BLEU 점수 계산
            bleu = sentence_bleu([orig.split()], aug.split())
            bleu_scores.append(bleu)
            
            # ROUGE 점수 계산
            try:
                rouge = self.rouge.get_scores(aug, orig)[0]
                rouge_scores.append(rouge['rouge-l']['f'])
            except:
                continue
                
        return {
            'mean_bleu': np.mean(bleu_scores),
            'mean_rouge_l': np.mean(rouge_scores)
        }
    

    def print_test_results(self, results: dict):
        """
        테스트 결과 출력
        Args:
            results (dict): 테스트 결과
        """
        print("\n=== 증강 데이터 품질 테스트 결과 ===")
        
        print("\n1. 기본 통계:")
        print(f"- 원본 데이터 크기: {results['basic_stats']['original_size']}")
        print(f"- 증강 데이터 크기: {results['basic_stats']['augmented_size']}")
        print(f"- 평균 길이 차이: {results['basic_stats']['length_difference']:.2f}")
        
        print("\n2. 의미적 유사도:")
        print(f"- 평균 유사도: {results['semantic_similarity']['mean_semantic_similarity']:.4f}")
        print(f"- 최소 유사도: {results['semantic_similarity']['min_semantic_similarity']:.4f}")
        print(f"- 최대 유사도: {results['semantic_similarity']['max_semantic_similarity']:.4f}")
        
        print("\n3. 텍스트 유사도:")
        print(f"- 평균 BLEU: {results['text_similarity']['mean_bleu']:.4f}")
        print(f"- 평균 ROUGE-L: {results['text_similarity']['mean_rouge_l']:.4f}")


# 사용 예시
if __name__ == "__main__":
    # 데이터 로드
    combined_df = pd.read_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/outputs/after_datas.csv')
    
    # 테스트 실행
    tester = AugmentorTester()
    results = tester.test_augmentation_quality(combined_df)
    tester.print_test_results(results)


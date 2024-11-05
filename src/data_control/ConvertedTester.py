from .noise_converter.NoiseConverter import NoiseConverter
from typing import List, Tuple
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge

class ConvertedTester:
    def __init__(self, converter: NoiseConverter):
        self.converter = converter

    def _calculate_bleu(self, original_sentences: List[str], converted_sentences: List[str]) -> float:
        """
        BLEU 점수를 계산하는 메소드

        Args:
            original_sentences: 원본 문장 리스트
            converted_sentences: 변환된 문장 리스트

        Returns:
            float: BLEU 점수
        """
        
        return corpus_bleu([[sentence] for sentence in original_sentences], converted_sentences)
    
    def _calculate_rouge(self, original_sentences: List[str], converted_sentences: List[str]) -> float:
        """
        ROUGE 점수를 계산하는 메소드

        Args:
            original_sentences: 원본 문장 리스트
            converted_sentences: 변환된 문장 리스트

        Returns:
            float: ROUGE 점수
        """
        rouge = Rouge()
        scores = rouge.get_scores(converted_sentences, original_sentences, avg=True)
        return scores['rouge-l']['f']
    
    def _calcuate_accuracy(self, original_sentences: List[str], converted_sentences: List[str]) -> float:
        """
        정확도를 계산하는 메소드

        Args:
            original_sentences: 원본 문장 리스트
            converted_sentences: 변환된 문장 리스트

        Returns:
            float: 정확도
        """
        correct_count = 0
        for original, converted in zip(original_sentences, converted_sentences):
            if original == converted:
                correct_count += 1
        return correct_count / len(original_sentences)
        
    def test(self, sentences: List[str]) -> Tuple[float, float, float]:
        """
        주어진 문장들에 대해 변환기의 성능을 테스트하는 메소드

        Args:
            sentences: 테스트할 문장 리스트

        Returns:
            float: 변환기의 성능
        """
        converted_sentences = self.converter.convert(sentences)
        bleu_score = self._calculate_bleu(sentences, converted_sentences)
        rouge_l_score = self._calculate_rouge(sentences, converted_sentences)
        accuracy_score = self._calcuate_accuracy(sentences, converted_sentences)
        
        print(f"BLEU: {bleu_score}, ROUGE-L: {rouge_l_score}, Accuracy: {accuracy_score}")

        return bleu_score, rouge_l_score, accuracy_score
from src.data_control.noise_converter.NoiseConverter import NoiseConverter
from typing import List, Tuple
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
from src.data_control.noise_generator.NoiseGeneratorASCII import NoiseGeneratorASCII
import pandas as pd


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
        
    def test(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        주어진 문장들에 대해 변환기의 성능을 테스트하는 메소드

        Args:
            df: 변환 전 문장들이 담긴 DataFrame

        Returns:
            float: 변환기의 성능
        """
        sentences = df['text'].tolist()
        noised_sentences_df = NoiseGeneratorASCII().generate(df)
        converted_sentences = self.converter.convert(noised_sentences_df)['text'].tolist()
        
        bleu_score = self._calculate_bleu(sentences, converted_sentences)
        rouge_l_score = self._calculate_rouge(sentences, converted_sentences)
        accuracy_score = self._calcuate_accuracy(sentences, converted_sentences)
        
        print(f"BLEU: {bleu_score}, ROUGE-L: {rouge_l_score}, Accuracy: {accuracy_score}")

        return bleu_score, rouge_l_score, accuracy_score
    
if __name__ == "__main__":
    from src.data_control.noise_converter.NCGemma2 import NCGemma2

    model_ids = ['rtzr/ko-gemma-2-9b-it', 'HumanF-MarkrAI/Gukbap-Gemma2-9B']
    prompts = ['prompts/gemma-converter.json', 'prompts/gukbap-gemma-converter.json']

    sample_df = pd.read_csv('data/cleaned.csv')
    sample_df = sample_df[sample_df['noise'] == 0].sample(100, random_state=42)

    for i, model_id in enumerate(model_ids):
        print(f"Testing {model_id}")
        converter = NCGemma2(model_id, prompt= prompts[i])
        tester = ConvertedTester(converter)
        tester.test(sample_df)

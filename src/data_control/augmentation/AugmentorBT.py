# BT

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Dataset
from googletrans import Translator
import time
import deepl
from Augmentor import Augmentor

class BackTranslator(Augmentor):
    def __init__(self, type="google", loop=1, lang="en", batch_size=16, deepl_api_key=None):
        """
        BackTranslator 초기화
        Args:
            type (str): 번역기 종류 ("google" 또는 "deepl")
            loop (int): 역번역 반복 횟수
            lang (str): 중간 번역 언어 코드
            batch_size (int): 배치 크기
            deepl_api_key (str): DeepL API 키
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.type = type
        self.loop = loop
        self.lang = lang
        self.batch_size = batch_size
        self.deepl_api_key = deepl_api_key
        
        
        # 번역기 초기화
        if self.type == "google":
            self.translator = Translator()
        elif self.type == "deepl":
            if not deepl_api_key:
                raise ValueError("DeepL API를 사용하려면 API 키가 필요합니다.")
            self.translator = deepl.Translator(deepl_api_key)
        
        print(f"Using device: {self.device}")
        print(f"Translator: {self.type}")
        print(f"Target language: {self.lang}")
        print(f"Loop count: {self.loop}")

    def _chunk_batch(self, iterable):
        """데이터를 배치 크기로 분할"""
        length = len(iterable)
        result = []
        for ndx in range(0, length, self.batch_size):
            result.append(iterable[ndx : min(ndx + self.batch_size, length)])
        return result

    def translate_text(self, text, src_lang, dest_lang):
        """단일 텍스트 번역"""
        if self.type == "google":
            translated = self.translator.translate(text, src=src_lang, dest=dest_lang).text
            time.sleep(0.5)
            return translated
        elif self.type == "deepl":
            try:
                # DeepL의 언어 코드 매핑
                source_lang_map = {
                    'ko': 'ko',      # source는 소문자 사용
                    'en': 'en',      
                    'ja': 'ja'       
                }
                
                target_lang_map = {
                    'ko': 'KO',      # target은 대문자 사용
                    'en': 'EN-US',   # 영어는 EN-US 형식 사용
                    'ja': 'JA'       # target은 대문자 사용
                }
                
                result = self.translator.translate_text(
                    text,
                    source_lang=source_lang_map[src_lang],
                    target_lang=target_lang_map[dest_lang]
                )
                time.sleep(0.5)
                return result.text
            except Exception as e:
                print(f"DeepL 번역 중 오류 발생: {e}")
                return text

    def _back_translate(self, batch):
        """배치 데이터에 대한 역번역 수행"""
        translated_texts = []
        for text in batch:
            try:
                current_text = text
                for _ in range(self.loop):
                    # 한국어 => 대상 언어
                    intermediate_text = self.translate_text(current_text, 'ko', self.lang)
                    # 대상 언어 => 한국어
                    current_text = self.translate_text(intermediate_text, self.lang, 'ko')
                translated_texts.append(current_text)
            except Exception as e:
                print(f"번역 중 오류 발생: {e}")
                translated_texts.append(text)
        return translated_texts

    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터셋 증강 수행
        Args:
            df (pd.DataFrame): augmentation을 적용할 dataframe
        Returns:
            pd.DataFrame: augmentation이 적용된 dataframe
        """
        target_ids = df['ID']
        target_docs = df['text']
        target_labels = df['target']

        # 배치 처리 및 번역
        batches = self._chunk_batch(target_docs)
        augmented_text = []

        print("Back Translation 시작...")
        for batch in tqdm(batches):
            augmented_text.extend(self._back_translate(batch))

        # 증강된 데이터셋 생성
        last_id_num = int(target_ids.iloc[-1].split('_')[-1])
        new_ids = [f"ynat-v1_train_{str(i).zfill(5)}" 
                  for i in range(last_id_num + 1, last_id_num + len(target_ids) + 1)]
        
        augmented_df = pd.DataFrame({
            'ID': new_ids,
            'text': augmented_text,
            'target': target_labels
        })

        # 원본 데이터와 병합하여 반환
        return pd.concat([df, augmented_df], ignore_index=True)

def test_augmentation():
    """테스트를 위한 실행 함수"""
    print("Back Translation 테스트 시작")

    # 테스트 데이터셋 로드
    test_df = pd.read_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train.csv')
    

    # 테스트 케이스 1: Google 번역, 영어
    print("\n테스트 1: Google 번역 (한국어 -> 영어 -> 한국어)")
    bt_google = BackTranslator(
        type="google",
        loop=1,
        lang="en",
        batch_size=16
    )
    augmented_df = bt_google.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_google_en_1loop.csv', 
                        index=False)
    
    # 테스트 케이스 2: Google 번역, 일본어
    print("\n테스트 2: Google 번역 (한국어 -> 일본어 -> 한국어)")
    bt_google_ja = BackTranslator(
        type="google",
        loop=1,
        lang="ja",
        batch_size=16
    )
    augmented_df = bt_google_ja.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_google_ja_1loop.csv', 
                        index=False)
    

    DEEPL_API_KEY = ""  # DeepL API 키 입력
    # 테스트 케이스 3: DeepL 번역, 영어
    print("\n테스트 3: DeepL 번역 (한국어 -> 영어 -> 한국어)")
    bt_deepl = BackTranslator(
        type="deepl",
        loop=1,
        lang="en",
        batch_size=16,
        deepl_api_key=DEEPL_API_KEY  # API 키 전달
    )
    augmented_df = bt_deepl.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_deepl_en_1loop.csv', 
                        index=False)
                      

    # 테스트 케이스 4: DeepL 번역, 일본어
    print("\n테스트 4: DeepL 번역 (한국어 -> 일본어 -> 한국어)")
    bt_deepl = BackTranslator(
        type="deepl",
        loop=1,
        lang="ja",
        batch_size=16,
        deepl_api_key=DEEPL_API_KEY
    )
    augmented_df = bt_deepl.augment(test_df)
    augmented_df.to_csv('/data/ephemeral/home/ksw/level2-nlp-datacentric-nlp-01/data/train_augmented_deepl_ja_1loop.csv', 
                        index=False)
    

if __name__ == "__main__":
    test_augmentation()

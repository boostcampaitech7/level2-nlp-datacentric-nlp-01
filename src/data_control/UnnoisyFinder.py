import pandas as pd
import nltk.translate.bleu_score as sentence_bleu
from transformers import AutoTokenizer
from typing import List

def turn_off_noisy(df: pd.DataFrame) -> pd.DataFrame:
    """noisy로 판단한 데이터 중에서 BLEU 스코어가 일정 값보다 높은 데이터를 찾아서 unnoisy로 변경합니다.

    Args:
        df (pd.DataFrame): `noisy` 컬럼을 가지고 있는 dataframe

    Returns:
        pd.DataFrame: noisy로 판단한 데이터 중 일부를 unnoisy로 변경한 dataframe
    """
    
    df_noised = df[df['noise'] == 1].dropna()
    
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')
    def tokenize(x: str) -> List[str]:
        return tokenizer.tokenize(x)
    
    def bleu_score(x: pd.Series) -> float:
        return sentence_bleu.sentence_bleu(
            [tokenize(x['original'])],
            tokenize(x['text'])
        )
    
    df_noised['bleu'] = df_noised.apply(bleu_score, axis=1)
    
    df['bleu'] = df_noised['bleu']
    df.loc[(df['bleu'] > 0.5), 'noise'] = 0
    df.loc[df['noise'] == 0, 'text'] = df['original'] 
    df.drop(columns=['bleu'], inplace=True)
    
    return df

def routine_unnoisy_finder():
    df = pd.read_csv('data/combined_clean_train.csv')
    print(df[df['noise'] == 1].shape[0], df[df['noise'] == 0].shape[0])
    df = turn_off_noisy(df)
    print(df[df['noise'] == 1].shape[0], df[df['noise'] == 0].shape[0])
    
    df.to_csv('data/BLEU.csv', index=False)
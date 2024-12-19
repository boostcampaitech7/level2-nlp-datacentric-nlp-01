from .noise_detector import NoiseDetectorASCII, ASCIIwithoutSpace
from . import ModuleTester
import pandas as pd

def binary_search(class_name, df: pd.DataFrame, moe: int = 15) -> float:
    """Detector 클래스를 받아 적절한 threshold를 이진 탐색으로 찾는다.
    2800개의 데이터를 1200 : 1600 으로 나눌 수 있는 threshold를 찾는다. 

    Args:
        class_name (_type_): 
        df (pd.DataFrame): threshold를 찾을 데이터
        moe (int, optional): margin of error. 1200으로 정확히 나누어 떨어지지 못할 수 있으므로 오차범위를 허용한다. Defaults to 15.

    Returns:
        float: 적절한 threshold
    """
    high = 1.0
    low = 0.0
    
    detector = class_name()
    df['score'] = df.apply(lambda x: detector._noise_score(x), axis=1)
    
    mid = (high + low) / 2
    while not (1200 - moe <= df[df['score'] < mid].shape[0] and df[df['score'] < mid].shape[0] <= 1200 + moe):
        num_unnoised = df[df['score'] < mid].shape[0]
        
        if num_unnoised < 1200 - moe:
            low = mid
        else:
            high = mid
        print(f"low: {low}, high: {high}, mid: {mid}, num_unnoised: {num_unnoised}")
        mid = (high + low) / 2
    
    return mid

def test_multi_detector():
    """Detector 클래스에 대한 적절한 threshold를 찾아 테스트하고 결과를 저장한다.
    """
    df = pd.read_csv('data/train.csv')
    
    classes = [NoiseDetectorASCII, ASCIIwithoutSpace]
    # [0.3828125, 0.2578125]
    
    for _, class_now in enumerate(classes):
        print(f"Name: {class_now.__name__}")
        threshold = binary_search(class_now, df)
        print(f"threshold: {threshold}")
        
        tester = ModuleTester(
            detector=class_now(ascii_threshold=threshold),
            output_path=f'data/outputs/{class_now.__name__}'
        )
        
        noised, unnoised = tester.test_detector(df)
        
        noised_sorted = noised.sort_values(by='score', ascending=True) # 오름차순
        unnoised_sorted = unnoised.sort_values(by='score', ascending=False) # 내림차순
        
        noised_sorted.head(100).to_csv(f'data/outputs/{class_now.__name__}/noised_sorted.csv')
        unnoised_sorted.head(100).to_csv(f'data/outputs/{class_now.__name__}/unnoised_sorted.csv')
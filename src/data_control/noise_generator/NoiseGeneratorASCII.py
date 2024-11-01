from src.data_control.noise_generator.NoiseGenerator import NoiseGenerator
import pandas as pd
import numpy as np
from typing import Optional 

class NoiseGeneratorASCII(NoiseGenerator):
    
    def __init__(self, ratio: Optional[float], 
                 min_ratio: float = 0.2, 
                 max_ratio: float = 0.8):
        """문자열의 일정 비율을 ASCII로 변환하여 noise를 추가한다.

        Args:
            ratio (Optional[float]): 비율을 고정한다. (min, max 무시)
            min_ratio (float, optional): 최소 비율. Defaults to 0.2.
            max_ratio (float, optional): 최대 비율. Defaults to 0.8.
        """
        
        if ratio is not None:
            self.min_ratio = ratio
            self.max_ratio = ratio
        else:
            self.min_ratio = min_ratio
            self.max_ratio = max_ratio
            
    def _add_noise(self, x: pd.Series) -> str:
        """문자열에 noise를 추가한다.

        Args:
            x (pd.Series): 행 데이터 

        Returns:
            str: noise가 추가된 문자열
        """
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        noise_len = int(len(x['text']) * ratio)
        noise_indicator = np.zeros(len(x['text']), dtype=bool)
        noise_indicator[np.random.choice(len(x['text']), noise_len, replace=False)] = True
        noised = ""
        for i, c in enumerate(x['text']):
            if noise_indicator[i]:
                noised += ''.join([chr(np.random.randint(32, 177))])
            else:
                noised += c
        x['text'] = noised
        return x
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 noise를 추가한다.

        Args:
            df (pd.DataFrame): noise가 없는 dataframe

        Returns:
            pd.DataFrame: noise가 추가된 dataframe
        """
        noised = df.applymap(self._add_noise)
        return noised
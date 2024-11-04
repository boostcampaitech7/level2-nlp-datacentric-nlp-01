from src.data_control.noise_generator.NoiseGenerator import NoiseGenerator
import pandas as pd
import numpy as np
from typing import Optional 

class NoiseGeneratorASCII(NoiseGenerator):
    
    def __init__(self, ratio: Optional[float] = None, 
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
            
    def _add_noise(self, x: str) -> str:
        """문자열에 noise를 추가한다.

        Args:
            x (str): 임의의 문자열

        Returns:
            str: noise가 추가된 문자열
        """
        ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        noise_len = int(len(x) * ratio)
        noise_indicator = np.zeros(len(x), dtype=bool)
        noise_indicator[np.random.choice(len(x), noise_len, replace=False)] = True
        noised = ""
        for i, c in enumerate(x):
            if noise_indicator[i]:
                # 32 ~ 63, 96 ~ 126
                rnd = np.random.randint(32 + 31)
                if rnd >= 32:
                    rnd += 96 - 32
                else:
                    rnd += 32
                noised += chr(rnd)
            else:
                noised += c
        return noised
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 noise를 추가한다.

        Args:
            df (pd.DataFrame): noise가 없는 dataframe

        Returns:
            pd.DataFrame: noise가 추가된 dataframe
        """
        df.loc[:, "text"] = df["text"].map(self._add_noise)
        return df
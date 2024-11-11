from abc import ABC, abstractmethod
import pandas as pd

class NoiseGenerator(ABC):
    
    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 noise를 추가한다.

        Args:
            df (pd.DataFrame): noise가 없는 dataframe

        Returns:
            pd.DataFrame: noise가 추가된 dataframe
        """
        pass
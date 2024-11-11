from abc import ABC, abstractmethod
import pandas as pd

class NoiseConverter(ABC):
    
    @abstractmethod
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 noise가 있다고 가정하고, 
           모든 행의 noise를 복원한다.

        Args:
            df (pd.DataFrame): 모든 행에 noise가 존재하는 dataframe

        Returns:
            pd.DataFrame: noise가 복원된 dataframe
        """
        pass
from abc import ABC, abstractmethod
import pandas as pd

class NoiseDetector(ABC):
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 있는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 있다고 판단된 데이터로만 이루어진 dataframe
        """
        pass
    
    @abstractmethod
    def detect_not(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 없는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 없다고 판단된 데이터로만 이루어진 dataframe
        """
        pass
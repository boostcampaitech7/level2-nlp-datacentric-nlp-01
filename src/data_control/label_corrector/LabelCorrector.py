from abc import ABC, abstractmethod
import pandas as pd

class LabelCorrector(ABC):

    @abstractmethod
    def train(self, df: pd.DataFrame) -> None:
        """label이 올바른 dataframe을 학습한다.
        Args:
            df (pd.DataFrame): noise가 있는 dataframe
        
        Returns:
            None (내부적인 변화만 있음 or 학습된 모델을 저장)
        """
        pass
    
    @abstractmethod
    def correct(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 noise가 없다고 가정하고, 
           모든 행의 label을 교정한다.

        Args:
            df (pd.DataFrame): noise가 없는 dataframe

        Returns:
            pd.DataFrame: label이 교정된 dataframe
        """
        pass
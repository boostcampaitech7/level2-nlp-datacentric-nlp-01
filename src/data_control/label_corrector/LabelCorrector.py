from abc import ABC, abstractmethod
import pandas as pd

class LabelCorrector(ABC):
    
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
from abc import ABC, abstractmethod
import pandas as pd

class Augmentor(ABC):
        
    @abstractmethod
    def augment(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 모든 행에 augmentation을 적용한다.

        Args:
            df (pd.DataFrame): augmentation을 적용할 dataframe
    
        Returns:
            pd.DataFrame: augmentation이 적용된 dataframe
        """
        pass
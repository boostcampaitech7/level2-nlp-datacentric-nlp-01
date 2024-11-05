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

    def get_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe의 각 행에 대한 ASCII 비율을 계산하여 반환합니다.

        Args:
            df (pd.DataFrame): 입력 dataframe

        Returns:
            pd.DataFrame: ASCII 비율이 포함된 dataframe
        """
        df = df.copy()
        df['ascii_ratio'] = df.apply(
            lambda x: sum([31 < ord(c) and ord(c) < 177 for c in x['text']]) / len(x['text']), 
            axis=1
        )
        return df

    def get_lowest_ascii_rows(self, df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
        """ASCII 비율이 가장 낮은 n개의 행을 반환합니다.

        Args:
            df (pd.DataFrame): 입력 dataframe
            n (int): 반환할 행의 수

        Returns:
            pd.DataFrame: ASCII 비율이 가장 낮은 n개의 행
        """
        df_with_ratios = self.get_ratio(df)
        return df_with_ratios.sort_values('ascii_ratio').head(n).drop('ascii_ratio', axis=1)    
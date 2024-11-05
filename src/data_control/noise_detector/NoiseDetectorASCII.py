from src.data_control.noise_detector.NoiseDetector import NoiseDetector
import pandas as pd

class NoiseDetectorASCII(NoiseDetector):
    
    def __init__(self, ascii_threshold: float = 0.25):
        """ASCII 문자의 비율이 threshold 이상인 데이터를 noise로 판단한다.

        Args:
            ascii_threshold (float): noise로 판단할 ASCII 문자의 비율
        """
        self.ascii_threshold = ascii_threshold
    
    def _is_noised(self, x: pd.Series) -> bool:
        """dataframe의 한 행이 noise인지 판단한다.

        Args:
            x (pd.Series): dataframe의 한 행

        Returns:
            bool: noise이면 True, 아니면 False
        """
        ascii_ratio = sum([(32 <= ord(c) and ord(c) <= 63) or (96 <= ord(c) and ord(c) <= 126) for c in x['text']]) / len(x["text"])
        return ascii_ratio >= self.ascii_threshold
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 있는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 있다고 판단된 데이터로만 이루어진 dataframe
        """
        noised = df[df.apply(lambda x: self._is_noised(x), axis=1)]
        return noised
    
    def detect_not(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 없는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 없다고 판단된 데이터로만 이루어진 dataframe
        """
        not_noised = df[df.apply(lambda x: not self._is_noised(x), axis=1)]
        return not_noised
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
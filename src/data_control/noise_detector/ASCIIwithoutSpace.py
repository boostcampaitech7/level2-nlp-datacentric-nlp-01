from src.data_control.noise_detector.NoiseDetector import NoiseDetector
import pandas as pd

class ASCIIwithoutSpace(NoiseDetector):
    
    def __init__(self, ascii_threshold: float = 0.25):
        """ASCII 문자의 비율이 threshold 이상인 데이터를 noise로 판단한다. (단, 공백을 모두 제거한 후 비율을 계산한다.)

        Args:
            ascii_threshold (float): noise로 판단할 ASCII 문자의 비율
        """
        self.ascii_threshold = ascii_threshold
    
    def _noise_score(self, x: pd.Series) -> float:
        """dataframe의 한 행으로부터 noise 점수를 계산한다.

        Args:
            x (pd.Series): dataframe의 한 행

        Returns:
            float: 얼마나 noise인지 나타내는 값
        """
        x["text"] = ''.join(x["text"].split())
        ascii_ratio = sum([(32 <= ord(c) and ord(c) <= 126) for c in x['text']]) / len(x["text"])
        return ascii_ratio
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 있는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 있다고 판단된 데이터로만 이루어진 dataframe
        """
        noised = df[df.apply(lambda x: self._noise_score(x) >= self.ascii_threshold, axis=1)]
        return noised
    
    def detect_not(self, df: pd.DataFrame) -> pd.DataFrame:
        """dataframe에서 noise가 없는 데이터를 탐지한다.

        Args:
            df (pd.DataFrame): 임의의 dataframe

        Returns:
            pd.DataFrame: noise가 없다고 판단된 데이터로만 이루어진 dataframe
        """
        not_noised = df[df.apply(lambda x: self._noise_score(x) < self.ascii_threshold, axis=1)]
        return not_noised
    
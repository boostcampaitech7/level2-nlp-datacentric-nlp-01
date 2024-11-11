from .noise_detector import NoiseDetector
import pandas as pd

class ValidationMaker:
    def __init__(self, detector: NoiseDetector):
        self.detector = detector
    
    def _df_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """입력으로 들어온 dataframe의 형식을 아래와 같이 변환합니다.
        - 'original' 컬럼을 'text' 컬럼으로 변경합니다.
        - 'text' 컬럼을 'converted' 컬럼으로 변경합니다.

        Args:
            df (pd.DataFrame): 'original', 'text', 'noise' 컬럼을 가지고 있는 dataframe

        Returns:
            pd.DataFrame: 변환된 dataframe
        """
        df = df.dropna()
        must_have_columns = ['original', 'text', 'noise']
        for col in must_have_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing: {df.columns}")
        
        df = df.rename(columns={"original": "text", "text": "converted"})
        return df

    def make_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """다음과 같은 방식으로 validation dataset을 생성합니다.
        - 'noise'가 1인 데이터 중 (즉, label이 올바른)
        - 'text' 칼럼에서 noise detector로 noise가 적당히 있는 데이터를 찾고
        - 그 중 일부를 validation dataset으로 생성합니다.

        Args:
            df (pd.DataFrame): 'text', 'noise' 컬럼을 가지고 있는 dataframe 
                               (_df_convert로 변환된 dataframe)

        Returns:
            pd.DataFrame: validation dataset
        """
        
        df_noised_by_llm = df[df['noise'] == 1]
        df_noised_by_llm['score'] = df_noised_by_llm.apply(
            lambda x: self.detector._noise_score(x), axis=1
        )
        
        df_noised_by_llm = df_noised_by_llm[(df_noised_by_llm['score'] <= 0.4) & (df_noised_by_llm['score'] > 0.3)]
        result = df_noised_by_llm.groupby('target', group_keys=False).apply(lambda x: x.nsmallest(10, 'score'))
        
        return result
    
def make_validation():
    from .noise_detector import ASCIIwithoutSpace
    
    detector = ASCIIwithoutSpace(ascii_threshold=0.15625)
    validation_maker = ValidationMaker(detector)
    
    df = pd.read_csv('data/combined_clean_train.csv')
    df = validation_maker._df_convert(df)
    validation_df = validation_maker.make_validation(df)
    
    validation_df.to_csv('data/validation.csv', index=False)
    

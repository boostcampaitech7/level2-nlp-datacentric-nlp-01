import pandas as pd
from typing import Tuple

def filter_continuous_hangeul(df: pd.DataFrame, lim: Tuple[int, int]) -> pd.DataFrame:
    """데이터프레임에서 연속된 한글이 가장 적게 나오는 row를 삭제합니다.
    
    Args:
        df (pd.DataFrame): 필터링하기 위한 데이터프레임
        lim (Tuple[int, int]): 연속된 한글이 lim[0]개, 그 개수가 lim[1]개보다 적으면 삭제합니다.
    
    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    
    def count(x: str) -> Tuple[int, int]:
        """연속된 한글의 개수를 반환합니다.
        
        Args:
            x (str): _description_
        
        Returns:
            int: _description_
        """
        count = 0
        max_count = 0
        max_count_count = 0
        for c in x:
            if '가' <= c and c <= '힣':
                count += 1
                if count > max_count:
                    max_count = count
                    max_count_count = 1
                elif count == max_count:
                    max_count_count += 1
            else:
                count = 0
        return (max_count, max_count_count)
    
    df['count'] = df['original'].apply(count)
    df = df[df['count'] > lim]
    # sorted_df = df.sort_values(by='count')

    return df

def routine_continuous_hangeul():
    df = pd.read_csv('data/combined_clean_train.csv')
    df = filter_continuous_hangeul(df, (2,1))
    df.to_csv('data/filter_continous_hangeul.csv', index=False)

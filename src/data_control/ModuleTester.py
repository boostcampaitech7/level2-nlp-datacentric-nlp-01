from src.data_control.noise_generator.NoiseGenerator import NoiseGenerator
from src.data_control.noise_detector.NoiseDetector import NoiseDetector
from src.data_control.noise_converter.NoiseConverter import NoiseConverter
from src.data_control.label_corrector.LabelCorrector import LabelCorrector
from src.data_control.augmentation.Augmentor import Augmentor
from typing import Optional, Tuple
import pandas as pd
import os

class ModuleTester:
    
    def __init__(self,
                    generator: Optional[NoiseGenerator] = None,
                    detector: Optional[NoiseDetector] = None,
                    corrector: Optional[LabelCorrector] = None,
                    convertor: Optional[NoiseConverter] = None,
                    augmentor: Optional[Augmentor] = None,
                    aug_after_convert: bool = True,
                    aug_after_correct: bool = True,
                    is_mini: bool = False,
                    save_output: bool = True,   
                    output_path: str = "data/outputs/"
                ):
        """Data Control 폴더의 각 모듈을 통합하여 테스트하는 Class

        Args:
            generator (Optional[NoiseGenerator], optional): 노이즈 생성기. Defaults to None.
            detector (Optional[NoiseDetector], optional): 노이즈 감지기. Defaults to None.
            corrector (Optional[LabelCorrector], optional): 레이블 교정기. Defaults to None.
            convertor (Optional[NoiseConverter], optional): 레이블 변환기. Defaults to None.
            augmentor (Optional[Augmentor], optional): 데이터 증강기. Defaults to None.
            aug_after_convert (bool, optional): (noisy) 레이블을 변환한 후에 증강을 시도합니다. 레이블 교정기가 학습할 데이터의 양이 늘어납니다. Defaults to True.
            aug_after_correct (bool, optional): (unnoisy) 레이블을 교정한 후에 증강을 시도합니다. Defaults to True.
            is_mini (bool, optional): 데이터의 개수를 일부만 남깁니다. 연산이 빨리질 수도 있습니다. Defaults to False.
            save_output (bool, optional): dataframe 계산 결과를 csv로 저장합니다. Defaults to True.
            output_path (str, optional): 계산 결과를 어떤 디렉토리에 저장할지 설정합니다. Defaults to "outputs/".
        """
        
        
        self.generator = generator
        self.detector = detector
        self.corrector = corrector
        self.convertor = convertor
        self.augmentor = augmentor
        
        self.aug_after_convert = aug_after_convert
        self.aug_after_correct = aug_after_correct
        
        self.is_mini = is_mini
        self.num_mini_rows = 128
        
        self.save_output = save_output
        self.output_path = output_path
        
    def _save_if_can(self, df: pd.DataFrame, name: str) -> None:
        """dataframe을 csv로 저장합니다. (save_output가 True일 때만)

        Args:
            df (pd.DataFrame): 저장할 dataframe
            name (str): 저장될 파일의 이름
        """
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path)
        if self.save_output:
            path = os.path.join(self.output_path, name)
            df.to_csv(path, index=False)
        
    def _cut_mini(self, df: pd.DataFrame, num_rows: Optional[int] = None) -> pd.DataFrame:
        """dataframe의 개수를 일부만 남깁니다.

        Args:
            df (pd.DataFrame): 잘라 낼 dataframe
            num_rows (Optional[int], optional): 몇개나 남길지. (None일시 내부적으로 지정된 값 사용) Defaults to None.

        Returns:
            pd.DataFrame: 잘라 낸 dataframe
        """
        if num_rows is None:
            num_rows = self.num_mini_rows
        return df.sample(num_rows)
    
    def test_detector(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        """Detector를 테스트합니다.
        detector가 설정되어 있지 않으면 모든 데이터에 노이즈가 없다고 판단합니다.

        Args:
            df (pd.DataFrame): 테스트할 dataframe

        Returns:
            Tuple[Optional[pd.DataFrame], pd.DataFrame]: noise가 있는 데이터, noise가 없는 데이터
        """
        if self.detector is not None:
            noisy, unnoisy = self.detector.detect(df), self.detector.detect_not(df)
            self._save_if_can(noisy, "noisy.csv")
            self._save_if_can(unnoisy, "unnoisy.csv")
            return noisy, unnoisy
        else:
            print("Detector is not set. Skip testing detector.")
            return None, df
            
    def test_generator(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generator를 테스트합니다.
        generator가 설정되어 있지 않으면 노이즈 생성을 시도하지 않습니다.

        Args:
            df (pd.DataFrame): 테스트할 dataframe

        Returns:
            Optional[pd.DataFrame]: 노이즈가 생성된 dataframe, generator가 설정되어 있지 않으면 None
        """
        if self.generator is not None:
            noisy_gen = self.generator.generate(df)
            self._save_if_can(noisy_gen, "noisy_gen.csv")
            return noisy_gen
        else:
            print("Generator is not set. Skip testing generator.") 
            return None
            
    def test_converter(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Converter를 테스트합니다.

        Args:
            df (pd.DataFrame): 테스트할 dataframe

        Returns:
            Optional[pd.DataFrame]: 노이즈를 복원한 dataframe, convertor가 설정되어 있지 않으면 None
        """
        if self.convertor is not None:
            converted = self.convertor.convert(df)
            self._save_if_can(converted, "converted.csv")
            return converted
        else:
            print("Convertor is not set. Skip testing convertor.")
            return None
            
    def test_corrector(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Corrector를 테스트합니다.

        Args:
            df_train (pd.DataFrame): Corrector를 학습할 dataframe
            df_test (pd.DataFrame): 테스트할 dataframe

        Returns:
            Optional[pd.DataFrame]: 레이블이 교정된 dataframe, corrector가 설정되어 있지 않으면 None
        """
        if self.corrector is not None:
            self.corrector.train(df_train)
            corrected = self.corrector.correct(df_test)
            self._save_if_can(corrected, "corrected.csv")
            return corrected
        else:
            print("Corrector is not set. Skip testing corrector.")
            return None
        
    def test_augmentor(self, df: pd.DataFrame, file_name: str = "augment.csv") -> Optional[pd.DataFrame]:
        """Augmentor를 테스트합니다.

        Args:
            df (pd.DataFrame): 테스트할 dataframe
            file_name (str, optional): 저장될 파일의 이름. (증강은 여러 번 일어나므로 다르게 저장할 수 있도록 설정) 
                                       Defaults to "augment.csv".

        Returns:
            Optional[pd.DataFrame]: 증강한 dataframe, augmentor가 설정되어 있지 않으면 None
        """
        if self.augmentor is not None:
            return self.augmentor.augment(df)
        else:
            print("Augmentor is not set. Skip testing augmentor.")
            return None
        
    def _augmentation(self, condition: bool, df: pd.DataFrame, file_name: str = "augment.csv") -> pd.DataFrame:
        """조건에 따라 증강을 시도합니다.

        Args:
            condition (bool): 증강을 시도할지 여부
            df (pd.DataFrame): 증강할 dataframe
            file_name (str, optional): 저장될 파일의 이름. Defaults to "augment.csv".

        Returns:
            pd.DataFrame: 증강된 dataframe
        """
        if condition:
            df_auged = self.test_augmentor(df)
            if df_auged is None:
                print("Caution: Augmentor is not set. Use unaugmented data.")
                return df
            else:
                self._save_if_can(df_auged, file_name)
                if self.is_mini:
                    return self._cut_mini(df_auged, num_rows=self.num_mini_rows*2)
                return df_auged
        else:
            return df
    
    def test(self, df: pd.DataFrame):
        """모든 모듈을 테스트합니다.

        Args:
            df (pd.DataFrame): 테스트할 dataframe

        Returns:
            Tuple[
                Tuple[
                    Tuple[pd.DataFrame, pd.DataFrame] : detector의 결과 (noise가 있는 데이터, noise가 없는 데이터) 
                    Tuple[pd.DataFrame, pd.DataFrame] : generator의 결과 (noise가 없는 데이터, noise가 추가된 데이터)
                    Tuple[pd.DataFrame, pd.DataFrame] : converter의 결과 (noise가 있는 데이터, noise가 변환된 데이터)
                    Tuple[pd.DataFrame, pd.DataFrame] : corrector의 결과 (noise가 없는 데이터, noise가 교정된 데이터)
                    Tuple[pd.DataFrame, pd.DataFrame] : augmentor의 결과 (원본 데이터, 증강된 데이터)
                ], pd.DataFrame                       : 최종 dataframe
            ]
        """
        df_noised, df_unnoised = self.test_detector(df)
        
        if self.is_mini:
            df_noised = self._cut_mini(df_noised)
            df_unnoised = self._cut_mini(df_unnoised)
        
        df_added = self.test_generator(df_unnoised)
        if df_noised is None:
            print("Caution: Detector is not set. Label of noised data may be incorrect.")
            df_noised = df_added
        
        df_converted = self.test_converter(df_noised)
        if df_converted is None:
            print("Caution: Convertor is not set. Use unconverted data.")
            df_converted = df_noised
        
        df_converted_before = df_converted
        df_converted = self._augmentation(self.aug_after_convert, df_converted)
            
        df_corrected = self.test_corrector(df_unnoised, df_converted)
        if df_corrected is None:
            print("Caution: Corrector is not set. Use uncorrected data.")
            df_corrected = df_converted
        
        df_corrected_before = df_corrected
        df_corrected = self._augmentation(self.aug_after_correct, df_corrected)
        df_converted = self._augmentation(self.aug_after_correct and not self.aug_after_convert, df_converted)

        df_concat = pd.concat([df_converted, df_corrected], axis=0)
        return (
                (df_noised, df_unnoised), # Result of detector
                (df_unnoised, df_added), # Result of generator
                (df_noised, df_converted_before), # Result of converter
                (df_unnoised, df_corrected_before), # Result of corrector
                (df, df_concat) # Result of augmentation
            ), df_concat # Result of all
        
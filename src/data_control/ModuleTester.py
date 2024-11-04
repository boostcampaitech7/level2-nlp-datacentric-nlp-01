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
                    output_path: str = "outputs/"
                ):
        
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
        if os.path.exists(self.output_path) is False:
            os.makedirs(self.output_path)
        if self.save_output:
            path = os.path.join(self.output_path, name)
            df.to_csv(path, index=False)
        
    def _cut_mini(self, df: pd.DataFrame, num_rows: Optional[int] = None) -> pd.DataFrame:
        if num_rows is None:
            num_rows = self.num_mini_rows
        return df.sample(num_rows)
    
    def test_detector(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
        if self.detector is not None:
            noisy, unnoisy = self.detector.detect(df), self.detector.detect_not(df)
            self._save_if_can(noisy, "noisy.csv")
            self._save_if_can(unnoisy, "unnoisy.csv")
            return noisy, unnoisy
        else:
            print("Detector is not set. Skip testing detector.")
            return None, df
            
    def test_generator(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.generator is not None:
            noisy_gen = self.generator.generate(df)
            self._save_if_can(noisy_gen, "noisy_gen.csv")
            return noisy_gen
        else:
            print("Generator is not set. Skip testing generator.") 
            return None
            
    def test_converter(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.convertor is not None:
            converted = self.convertor.convert(df)
            self._save_if_can(converted, "converted.csv")
            return converted
        else:
            print("Convertor is not set. Skip testing convertor.")
            return None
            
    def test_corrector(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> Optional[pd.DataFrame]:
        if self.corrector is not None:
            self.corrector.train(df_train)
            corrected = self.corrector.correct(df_test)
            self._save_if_can(corrected, "corrected.csv")
            return corrected
        else:
            print("Corrector is not set. Skip testing corrector.")
            return None
        
    def test_augmentor(self, df: pd.DataFrame, file_name: str = "augment.csv") -> Optional[pd.DataFrame]:
        if self.augmentor is not None:
            return self.augmentor.augment(df)
        else:
            print("Augmentor is not set. Skip testing augmentor.")
            return None
        
    def _augmentation(self, condition: bool, df: pd.DataFrame, file_name: str = "augment.csv") -> pd.DataFrame:
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
    
    def test(self, df: pd.DataFrame) -> None:
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
        
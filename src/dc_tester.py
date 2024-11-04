from .data_control import ModuleTester, augmentation, label_corrector, noise_converter, noise_detector, noise_generator
import pandas as pd

def module_test():
    tester = ModuleTester(
        generator=noise_generator.NoiseGeneratorASCII(),
        detector=noise_detector.NoiseDetectorASCII(),
        # corrector=label_corrector.LabelCorrector(),
        # convertor=noise_converter.NoiseConverter(),
        # augmentor=augmentation.Augmentor(),
    )
    
    df = pd.read_csv('data/train.csv')
    
    results, df_new = tester.test(df)
    
    result_detector, result_generator, result_converter, result_corrector, result_augmentor = results
    
    det_noisy, det_unnoisy = result_detector 
    gen_bef, gen_aft = result_generator
    conv_bef, conv_aft = result_converter
    corr_bef, corr_aft = result_corrector
    augm_bef, augm_aft = result_augmentor
    
if __name__ == "__main__":
    module_test()
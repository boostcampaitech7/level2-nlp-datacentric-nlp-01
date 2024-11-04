from src.train import trainer
from src.evaluate import evaluate_and_save
from src.dc_tester import module_test 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add = parser.add_argument('--data_control', action='store_true', help='Run data control module test')
    args = parser.parse_args()    

    if args.data_control:
        module_test()
    else:
        trainer.train()
        evaluate_and_save()  # 학습 후 테스트 데이터 예측 및 결과 저장

from src.train import trainer
from src.evaluate import evaluate_and_save

if __name__ == '__main__':
    trainer.train()
    evaluate_and_save()  # 학습 후 테스트 데이터 예측 및 결과 저장

from AugmentorEDA import EasyDataAugmentation
import pandas as pd

# 증강기 초기화
augmentor = EasyDataAugmentation(
    methods=['RI'],  # 원하는 방법만 선택
    morpheme_analyzer="Mecab",
    p=0.3,
    repetition=1
)

# CSV 파일 처리
input_path = "/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/data/train.csv"
output_path = "/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/data/augmented_SR_train.csv" 

input_data = pd.read_csv(input_path)
aug_data = augmentor.augment(input_data, text_column='text')


# Save the augmented data to a new CSV file
aug_data.to_csv(output_path, index=False)
print(f"Augmented data saved to {output_path}")
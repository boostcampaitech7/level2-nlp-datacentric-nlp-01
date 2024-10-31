# Level2-nlp-datacentric-project

## 0. Workflow
<img src= "https://github.com/user-attachments/assets/0f130239-cf61-4fce-8ffc-959f7c2ce12c" width="700" height="300" />

## 1. 파일구성
```
level2-nlp-datacentric-nlp-01/
|-- README.md
|-- data
|-- etc
|   `-- image_work_flow.png
|-- main.py                                     # 메인 실행 파일
|-- requirements.txt                            # 필요한 패키지 목록
|-- src
   |-- config.py
   |-- data_control
   |   |-- augmentation                         # 데이터 증강 모듈
   |   |-- label_corrector                      # 레이블 교정 모듈
   |   |-- noise_converter                      # 노이즈 변환 모듈
   |   |-- noise_detector                       # 노이즈 탐지 모듈
   |   `-- noise_generator                      # 노이즈 생성 모듈 (prompting 용도)
   |       
   |-- dataset.py                               # 데이터셋 로딩 및 처리 모듈
   |-- evaluate.py                              # 모델 평가 모듈
   |-- model.py                                 # 모델 정의
   |-- train.py                                 # 학습
   `-- utils.py                                 # compute_metrics 등 utils 폴더

```

## 2. 실행방법

CLI 창에서
`python main.py`
라고 치면 훈련 및 평가(output.csv 저장)까지 자동으로 진행됩니다.

# 프로젝트 보고서: Data-Centric Topic Classification

---

## 1. 팀 소개 및 프로젝트 개요
### 주제: 데이터 중심 주제 분류
뉴스 헤드라인을 분석해 주제를 분류하는 데 있어, **모델 변경 없이 데이터 가공만으로 성능을 높이는** 방법을 연구. 

별도의 작동 코드는 없으나, 전처리 모듈들과 증강 모듈들을 확실히 클래스화 하여 이후 재사용성을 높임.
구체적으로 본 프로젝트에서는 텍스트와 라벨 노이즈를 식별하고 제거하는 방법, 데이터 증강, 텍스트 정제 등의 접근 방식을 
**Data-Centric**하게 적용하였음.

### 팀원 및 역할
|팀원|역할|
|----|----|
|박준성|**모듈화 가이드라인 작성 및 유지보수**, **Noise Detector 구현**(rule-based, BLEU 점수 기반 LLM 결과 재판별), **데이터셋 성능 평가를 위한 Validation 데이터셋 확보 및 실험 관리** |
|이재백|**Noise Converter 성능 평가 구현**(BLEU, ROUGE-L 사용),  **라벨 노이즈 처리**(Cleanlab)|
|강신욱|**데이터 증강** (Back Translation) 및 **증강 데이터 평가 및 실험**|
|홍성균|**데이터 EDA**(텍스트 길이, 형태소 분석, 군집화 등), **데이터 정제 및 증강**(LLM, Rule-based), **Streamlit을 통한 데이터 분석 공유**|
|백승우|**데이터 시각화** (WordCloud), **Noise Converter 구현** (LLM, vLLM 도입 시도)|
|김정석|**EDA**(텍스트 길이, 문자군별 분포 분석 등)|

### 목표 및 평가 지표
- **목표**: Data-centric 관점에서 데이터의 품질을 개선하여 주제 분류 성능을 높이는 것.
- **평가 지표**:
  - **Macro F1**: 모델 성능의 핵심 지표.
  - **Accuracy**: 참고 지표로 활용.
  
### 데이터 구성
- **Train 데이터**: 2,800개 (1,600개는 텍스트 노이즈 포함, 1,000개는 라벨 노이즈 포함, 200개는 정상)
- **Test 데이터**:
  - Public: 15,000개 (ID, 텍스트만 공개)
  - Private: 15,000개

---

## 2. 프로젝트 진행
### 2.0 전체 워크플로우
프로젝트는 다음과 같다.

1. **NoiseDetector**: 텍스트 노이즈가 있는 데이터를 식별
2. **NoiseConverter**: 식별된 노이즈 데이터를 복원
3. **LabelCorrector**: 텍스트 노이즈가 있던 데이터의 라벨은 정확하다는 점을 활용하여 다른 데이터의 라벨을 교정
4. **DataAugmentor**: 정제된 데이터를 기반으로 증강 수행
### 2.1 텍스트 노이즈 제거
- **LLM 모델 선정 과정**:
  - LogicKor 등 리더보드에서 '공개' 모델들을 조사
  - 10B 이하 모델들 중 추론, 작문, 이해, 문법 성능이 우수한 모델들을 선별
  - 3개 모델 성능 비교 결과:
    - rtzr/ko-gemma-2-9b-it: BLEU 0.3336, ROUGE-L 0.2304
    - HumanF-MarkrAI/Gukbap-Gemma2-9B: BLEU 0.2493, ROUGE-L 0.1634
    - NCSOFT/Llama-VARCO-8B-Instruct: BLEU 0.2368, ROUGE-L 0.1704

### 2.2 라벨 노이즈 처리
- **Cleanlab 라이브러리 활용**:
  - `Cleanlab`의 `find_label_issue` 메서드를 사용하여 self-confidence 값이 낮은 라벨을 노이즈로 판단.
  - **라벨 노이즈 탐지 결과**:
    - LLM으로 교정한 데이터셋을 사용한 경우 `f1: 0.8022` / 라벨 노이즈를 제거한 데이터셋을 사용한 경우 `f1: 0.7918`.
  - **라벨 노이즈 스를 전달
  4. 문법적 변화: 단어에 맞는 조사와 접속사 수정, 능동/수동태 전환 등
  5. 숫자 표현 변화: 숫자를 한글 또는 아라비아 숫자로 다양하게 표시
  6. 약어와 확장: 약어 사용 또는 풀어쓰기를 통해 표현 변화
  7. 문맥 고려: 한국의 사회적, 시사적 맥락에 맞게 변형
  8. 격식 유지: 원문에 맞는 격식을 유지한 공식적 표현 사용
  9. 관용표현 활용: 자연스럽게 어울리는 관용표현을 사용
  

---

## 3. 피드백 및 교훈
### 피드백
1. **프로그래밍 스타일의 제한**:
   - 객체 지향성을 도입하려 했으나, 코드 복잡성으로 인해 효율성이 감소. 데이터 중심 프로젝트에서는 소프트웨어 설계 원칙을 제한적으로 적용해야 함을 인지.
   
2. **데이터 버전 관리 미흡**:
   - 데이터셋 중복 훈련, 미완성 데이터셋 사용 등 관리 부족. 이를 개선하기 위해 DVC 등의 데이터 버전 관리 도구 사용 필요성 제기.

3. **LLM의 활용 부족**:
   - Rule-based 방법에 지나치게 의존하여, LLM의 잠재력을 충분히 발휘하지 못함.
   - 프롬프트 엔지니어링을 더 세밀히 설정하여 LLM을 적극 활용할 필요가 있음.

4. **기본 EDA 자동화 미흡**:
   - 데이터 길이, 라벨 균형 등을 자동화하여 EDA 수행이 가능하도록 템플릿화 부족. 이로 인해 데이터 정제 중 라벨 누락 등의 실수가 발생.

### 성과
- **모듈화**:
  - Noise Detector, Noise Converter, Label Corrector, Data Augmentor 등의 모듈화 작업을 통해 각기 다른 증강, 라벨 교정, 노이즈 제거 방식을 손쉽게 적용 가능하도록 구현.
  
- **협업 도구 활용**:
  - **Git**을 통한 개인 작업 및 Merge 세션 진행
  - **Notion**으로 실험 상황 및 서버 현황 공유.
  - **Zoom**을 통해 피드백 및 실시간 회의로 협업 강화.

---

## 4. 마무리
|| Accuracy     | Macro F-1     |
|---|--------|--------|
최종| **0.8317** | **0.8265** |
초기| 0.6107 | 0.6179 |

- **최종 데이터셋 구성**: 기존 2800개 데이터셋에서 약 3배 증가된 8,400개 학습 데이터 획득
- **모델 성능 향상**: 텍스트 노이즈 및 라벨 노이즈를 개선한 데이터셋을 바탕으로 LLM을 통해 증강을 진행하여 분류 성능을 향상시킬 수 있었음.

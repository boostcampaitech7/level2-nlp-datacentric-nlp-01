from src.data_control.noise_converter.NoiseConverter import NoiseConverter
import pandas as pd
from vllm import LLM, SamplingParams

class NoiseConverterLLM(NoiseConverter):
    def __init__(self, model_name: str = "NCSOFT/Llama-VARCO-8B-Instruct",
                 temperature: float = 1.0,
                 top_p: float = 1.0,
                 top_k: int = -1,):
        """vLLM을 활용하기 위한 초기화 작업

        Args:
            model_name (str, optional): 사용할 LLM모델 이름. Defaults to "NCSOFT/Llama-VARCO-8B-Instruct".
            temperature (float, optional): 모델이 다음 단어를 선택할 때의 랜덤성을 조절.
                                            창의적인 답변이 필요할 때는 값을 높이고, 일관성 있는 답변이 필요할 때는 낮춥니다.
                                            Defaults to 1.0.
            top_p (float, optional): 누적 확률이 p 이하가 될 때까지 상위 토큰 후보를 샘플링.
                                        p가 낮을 수록 상위 확률을 가진 토큰만 샘플링하여 더 자연스럽게 출력합니다.
                                        Defaults to 1.0.
            top_k (int, optional): k개의 상위 확률을 가진 토큰 중에서만 선택.
                                    특정 수의 상위 토큰에서만 샘플링하여 출력을 제어할 때 사용합니다.
                                    Defaults to -1.
        """
        self.model_name = model_name
        self.llm = LLM(model=model_name, dtype="float16")
        self.sampling_params = SamplingParams(temperature=temperature,
                                            top_p=top_p,
                                            top_k=top_k)
        
    def convert(self, df: pd.DataFrame, prompt: str) -> pd.DataFrame:
        """노이즈가 존재하는 dataframe을 받아 입력한 프롬프트로 노이즈를 복원하여 반환하는 function

        Args:
            df (pd.DataFrame): 모든 행에 noise가 존재하는 dataframe
            prompt (str): LLM의 입력으로 줄 프롬프트

        Returns:
            pd.DataFrame: noise가 복원된 dataframe
        """
        # 데이터프레임의 각 행을 반복하여 프롬프트 생성 및 텍스트 복구 실행
        prompts = []
        for text in df['text']:
            # prompt = f"문장의 일부분이 다음과 같이 아스키 코드로 대체되어 노이즈가 있어. 노이즈가 있는 문장: 한선]I 평정심U순위 싸움에3신경1쓰F,페이스를 %어요\n이 문장을 원래 문장으로 복구하면 다음과 같아. 복구된 문장: 한선수의 평정심 순위 싸움에 신경 쓰면 페이스를 잃어요\n그럼 이렇게 노이즈가 있는 문장을 복구해줘. 노이즈가 있는 문장: {text}\n복구된 문장: "
            prompt = f"{prompt}{text}"
            prompts.append(prompt)
        
        # LLM을 사용하여 텍스트 생성
        outputs = self.llm.generate(prompts, self.sampling_params)

        # 결과 저장을 위한 빈 리스트 생성
        results = []

        # 각 출력 결과를 반복하여 새로운 데이터프레임에 추가
        for i, output in enumerate(outputs):
            # original_text = df.loc[i, 'text']
            generated_text = output.outputs[0].text
            
            # 결과를 딕셔너리 형태로 리스트에 추가
            results.append({
                # 'Original text': original_text,
                'Generated text': generated_text
            })

        # 리스트를 데이터프레임으로 변환
        results_df = pd.DataFrame(results)
        return results_df
from .NoiseConverter import NoiseConverter
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)

import yaml
import os
import pandas as pd
from typing import Optional

class NCGoogleGenAI(NoiseConverter):
    def __init__(self, model_name: str = "gemini-1.5-flash",
                    system_prompt: str = (
                        "주어진 모든 문자열은 그 중 일부가 랜덤한 ASCII 문자로 대체되었습니다. 대체된 문자를 원래 문자로 복원하세요.\n"
                        "또한 다음과 같은 형식으로 답변을 생성해야 합니다: {{ \"Answer\": \"<복원된 문자열>\" }}"
                    ),
                    few_shots: Optional[pd.DataFrame] = None
                 ):
        self.set_secret_key()
        self.system_prompt = system_prompt
        self.llm = ChatGoogleGenerativeAI(model = model_name)
        self.few_shots = few_shots
        
    def set_secret_key(self, file_name: str = "./secrets.yaml"):
        """Google API Key를 환경변수에 저장한다.

        Args:
            file_name (str, optional): 불러올 파일 경로. Defaults to "secrets.yaml".
        """
        with open(file_name, 'r') as f:
            secrets = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["GOOGLE_API_KEY"] = secrets["GOOGLE_API_KEY"]
    
    def make_examples(self):
        examples = []
        if self.few_shots is None:
            return examples
        for _, row in self.few_shots.iterrows():
            examples.append(
                {
                    "given_string": row["noised_text"],
                    "answer": row["text"]
                }
            )
        return examples
            
        
    def convert(self, df):
        system_msg_prompt = SystemMessagePromptTemplate.from_template(self.system_prompt)
        
        # example_prompt = PromptTemplate.from_template(
        #     "Given String:\n{given_string}\nAnswer:\n{answer}"
        # )
        
        # few_shot_prompt = FewShotPromptTemplate(
        #     suffix = "Given String:\n{given_string}\nAnswer:\n",
        #     examples = self.make_examples(),
        #     example_prompt = example_prompt,
        #     input_variables = ["given_string"],
        # )
        
        example_messages = []
        
        for example in self.make_examples():
            human = HumanMessagePromptTemplate.from_template("Given String:\n{given_string}\n").format(given_string = example["given_string"])
            ai = AIMessagePromptTemplate.from_template("{{ \"Answer\": \"{answer}\" }}\n").format(answer = example["answer"])
            example_messages.append(human)
            example_messages.append(ai)
        
        suffix_tmp = HumanMessagePromptTemplate.from_template("Given String:\n{given_string}\n")
        
        chat_prompt = ChatPromptTemplate(
            messages = [system_msg_prompt] + example_messages + [suffix_tmp],
        )
        
        parser = JsonOutputParser()
        
        chat_prompt = chat_prompt.partial(
            format_instance = parser.get_format_instructions()
        )
        
        chain = chat_prompt | self.llm | parser
        
        df = df.copy()
        print(self.make_examples())
        x = []
        for _, row in df.iterrows():
            # print(chain.invoke({"given_string": row["text"]}))
            # print(row["text"])
            json = chain.invoke({"given_string": row["text"]})
            x.append(json["Answer"])
        df["text"] = x
        
        return df
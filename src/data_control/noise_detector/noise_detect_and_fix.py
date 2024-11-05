import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
from tqdm import tqdm


model_id = "rtzr/ko-gemma-2-9b-it"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

import pandas as pd
import re
from tqdm import tqdm  # tqdm 임포트

# Load the split dataset
# 각자 파일 명에 맞게 변경
input_file = "/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/src/data_control/augmentation/train_split_2.csv"
output_file = "clean_train_2.csv"
data = pd.read_csv(input_file)

# Placeholder for the cleaned data
cleaned_data = {
    "ID": [],
    "original": [],
    "text": [],
    "target": [],
    "noise": []
}

# Function to call the model and process each row's text
def clean_text_with_model(text, label):
    # Format the message to use as model input
    messages = [
    {
    "role": "system",
    "content": (
    "Guidelines:\n"
    "1. Context Understanding: Prioritize comprehension of each sentence’s main theme and intent to ensure natural, accurate Korean expression.\n\n"
    "2. Character Handling:\n"
    " - CRITICAL: Never convert Chinese characters (한자) to Hangul; Chinese characters are essential and not noise.\n"
    " - Remove special characters if they do not affect meaning or flow.\n"
    " - Convert numbers to Korean words only if necessary for context.\n\n"
    "3. Text Preservation and Restoration:\n"
    " - Keep the original text if grammatically correct and contextually appropriate.\n"
    " - For noise in the text, translate it into appropriate Korean syllables.\n"
    " - Restore proper nouns accurately to preserve intended meaning.\n\n"
    "4. Grammar and Structure:\n"
    " - Adjust particles, endings, or conjunctions only if clearly incorrect.\n"
    " - Preserve original sentence structure if it is sound, and apply minimal changes only when essential for clarity.\n\n"
    "5. Spacing and Formatting:\n"
    " - Correct only obvious spacing errors.\n"
    " - Make spacing adjustments only if they improve readability without altering the intended style.\n\n"
    "6. Noise and Content Evaluation:\n"
    " - Label noise as 1 if characters impair comprehension or clarity.\n"
    " - Do not mark Chinese characters as noise, even if translated into Hangul.\n"
    " - Maintain original stylistic elements unless they obstruct understanding.\n\n"
    "7. Contextual Consistency:\n"
    " - Use surrounding context to validate any modifications.\n"
    " - Differentiate between formal and informal writing styles, respecting genre and writing conventions.\n\n"
    "8. Korean Language Standards:\n"
    " - Ensure modifications adhere to Korean grammar rules.\n"
    " - Preserve intentional stylistic deviations from standard forms if they enhance meaning or style.\n\n"
    "Precautions:\n"
    " - Top Priority: Never convert or mark Chinese characters (한자) as noise.\n"
    " - When uncertain, keep the original text if comprehensible.\n"
    " - Handle loanwords carefully and preserve intentional stylistic elements.\n\n"
    "Objective: Carefully evaluate each text, making essential modifications only to improve comprehension or correct evident errors. All Chinese characters must remain in their original form, and intentional stylistic choices should be preserved. Aim to produce clear, natural Korean text that maintains the author’s voice and original intent."
    )},
    {"role": "user", "content": "text: '갯벌도 얼려버린^한파'\nlabel: 0"},
    {"role": "assistant", "content": "text: '갯벌도 얼려버린 한파'\nlabel: 0\nnoise: 1"},
    {"role": "user", "content": "text: '與 대표 선출, 친박·비박 구도 격화'\nlabel: 2"},
    {"role": "assistant", "content": "text: '與 대표 선출, 친박·비박 구도 격화'\nlabel: 2\nnoise: 0"},
    {"role": "user", "content": "text: 'cI그1[사령탑이 탐내C 선t는…이재성·이근호'\nlabel: 1"},
    {"role": "assistant", "content": "text: 'K리그 사령탑이 탐내는 선수는…이재성·이근호'\nlabel: 1\nnoise: 1"},
    {"role": "user", "content": "text: '새누리 원내대표 CL 화|는,여소야대·계파'\nlabel: 2"},
    {"role": "assistant", "content": "text: '새누리 원내대표 선출 앞두고 불거지는 여소야대·계파'\nlabel: 2\nnoise: 1"},
    {"role": "user", "content": "text: '감사원 수과학j구소M연구보다h워크>·강연'\nlabel: 3"},
    {"role": "assistant", "content": "text: '감사원 수리과학연구소, 연구보다 네트워크·강연'\nlabel: 3\nnoise: 1"},
    {"role": "user", "content": "text: '이통Y사f갤럭시탭Sr 오늘w터L사전예약 시작'\nlabel: 4"},
    {"role": "assistant", "content": "text: '이통3사, 갤럭시탭S7 오늘부터 사전예약 시작'\nlabel: 4\nnoise: 1"},
    {"role": "user", "content": "text: 'T시판 H성선물 하반기 p율·유가U망 세미나'\nlabel: 5"},
    {"role": "assistant", "content": "text: '한국거래소, 선물 하반기 금리·유가전망 세미나'\nlabel: 5\nnoise: 1"},
    {"role": "user", "content": "text: '與 김재수 해임건의안 제출에 옹졸한 정치보복성 공세종합'\nlabel: 2"},
    {"role": "assistant", "content": "text: '與 김재수 해임건의안 제출에 옹졸한 정치보복성 공세종합'\nlabel: 2\nnoise: 0"},
    {"role": "user", "content": "text: '중국의 해외투자 안전도I평가서 Z국 O위→=_위t추락'\nlabel: 6"},
    {"role": "assistant", "content": "text: '중국의 해외투자 안전도 평가서 미국 4위→14위 추락'\nlabel: 6\nnoise: 1"},
    {"role": "user", "content": "text: '박상R·w}욱 c 시B 6권 현대문학 *s시 즈 첫^출간'\nlabel: 5"},
    {"role": "assistant", "content": "text: '박상륭·최승욱 등 시인 6명 현대문학 시리즈 첫 출간'\nlabel: 5\nnoise: 1"},
    {"role": "user", "content": "text: '미스터 _샤인 S프로 부^…인물들 납y해'\nlabel: 0"},
    {"role": "assistant", "content": "text: '미스터 션샤인 OST 프로듀서 부부…인물들 납득해'\nlabel: 0\nnoise: 1"},
    {"role": "user", "content": "text: '월드컵 r안하|는 말/반%한G손흥' 울지c않으려고H..'\nlabel: 1"},
    {"role": "assistant", "content": "text: '월드컵 안하려는 말 반박한 손흥민 '울지 않으려고 했다''\nlabel: 1\nnoise: 1"},
    {"role": "user", "content": "text: '北0대! 김정*2향한 '성경쟁 무대된l릴레이r토론''\nlabel: 2"},
    {"role": "assistant", "content": "text: '北 20대 김정은 향한 충성경쟁 무대된 릴레이 토론'\nlabel: 2\nnoise: 1"},
    {"role": "user", "content": "text: '안경은%X: 보O기i 안되고…경찰관 공채u신체3준 논란'\nlabel: 3"},
    {"role": "assistant", "content": "text: '안경은 OK, 보청기는 안되고…경찰관 공채 신체기준 논란'\nlabel: 3\nnoise: 1"},
    {"role": "user", "content": "text: '이녹스 1분기 영업이익 17억원…76% 감소'\nlabel: 3"},
    {"role": "assistant", "content": "text: '이녹스 1분기 영업이익 17억원…76% 감소'\nlabel: 3\nnoise: 0"},
    {"role": "user", "content": "text: '위클리 스마트d스트레8E~N엔 휴,·면역강화엔 명상 과C적 l증'\nlabel: 4"},
    {"role": "assistant", "content": "text: '위클리 스마트폰 스트레스엔 휴대폰·면역강화엔 명상 과학적 입증'\nlabel: 4\nnoise: 1"},
    {"role": "user", "content": "text: '국토h 공동주택 관리 n우수0지에M서울 텐r힐 1단지'\nlabel: 5"},
    {"role": "assistant", "content": "text: '국토부 공동주택 관리 최우수단지에 서울 래미안 힐스테이트 1단지'\nlabel: 5\nnoise: 1"},
    {"role": "user", "content": "text: '이탈리아^독일~wGn난민구조선~억9…S6규정n등 위반'\nlabel: 6"},
    {"role": "assistant", "content": "text: '이탈리아·독일 NGO 난민구조선 억류…EU 규정 등 위반'\nlabel: 6\nnoise: 1"},
    {"role": "user", "content": f"text: '{text}'\nlabel: {label}"}]

    # Generate model output (replace with actual model call if running live)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Extract values from the output using regex
    cleaned_text = re.search(r"text:\s'([^']*)'", output).group(1) if re.search(r"text:\s'([^']*)'", output) else ""
    label = int(re.search(r"label:\s(\d+)", output).group(1)) if re.search(r"label:\s(\d+)", output) else ""
    noise = re.search(r"noise:\s([10])", output).group(1) if re.search(r"noise:\s([10])", output) else ""
    reason_match = re.search(r"reason:\s(.+)", output)
    reason = reason_match.group(1).strip() if reason_match else ""

    return cleaned_text, label, noise, reason

# Process each row in the data with tqdm progress bar and print every 10th row
for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
    text = row["text"]
    label = row["target"]

    # Clean the text with the model
    cleaned_text, cleaned_label, noise, reason = clean_text_with_model(text, label)

    # Add to the cleaned data dictionary
    cleaned_data["ID"].append(row["ID"])
    cleaned_data["original"].append(text)
    cleaned_data["text"].append(cleaned_text)
    cleaned_data["target"].append(cleaned_label)
    cleaned_data["noise"].append(noise)

    # Print every 10th row
    if (index + 1) % 10 == 0:
        print(f"Row {index + 1} - cleaned_text: {cleaned_text}, label: {cleaned_label}, noise: {noise}, reason: {reason}")

# Save cleaned data to new CSV
cleaned_df = pd.DataFrame(cleaned_data)
cleaned_df.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
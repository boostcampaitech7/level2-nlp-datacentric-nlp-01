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
model.eval()


# Load the split dataset
value_count_df = pd.read_csv('/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/src/EDA/value_counts.csv')
data = pd.read_csv("/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/error_label_server_2.csv")
output_file = "Trim_with_LLM_based_on_morphs_no_fewshot"

list_0 = list(value_count_df.target_0)[:30]
list_1 = list(value_count_df.target_1)[:30]
list_2 = list(value_count_df.target_2)[:30]
list_3 = list(value_count_df.target_3)[:30]
list_4 = list(value_count_df.target_4)[:30]
list_5 = list(value_count_df.target_5)[:30]
list_6 = list(value_count_df.target_6)[:30]

print(                f"""target 0: {list_0}
                target 1: {list_1}
                target 2: {list_2}
                target 3: {list_3}
                target 4: {list_4}
                target 5: {list_5}
                target 6: {list_6}""")

def clean_text_with_model(text, target):
    messages = [
    {
    "role": "system",
    "content": (f"""
                너는 뛰어난 분류기로써 0부터 6까지 해당하는 분류에 대해 교정해주는 역할을 아주 섬세하게 잘 해
                각 주제가 모호해 보일지라도 가장 그럴 듯한 하나의 분류를 제시하지
                그리고 각 분류의 핵심 단어는 이하와 같아
                target 0: {list_0}
                target 1: {list_1}
                target 2: {list_2}
                target 3: {list_3}
                target 4: {list_4}
                target 5: {list_5}
                target 6: {list_6}
                이제 너에게 text를 건넬테니 맞는 주제를 분류해내봐
                반드시 너는 target: 이 뒤를 이어서 대답해야해 숫자 단 하나로
                명심해 하나의 숫자(0,1,2,3,4,5,6) 중 하나로 대답해야해 핵심 단어들을 잘 고려해봐
                이제 바로 text를 건네줄게""")},
    {"role": "user", "content": f"text: {text}"}]

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
        max_new_tokens=10,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.8,
        top_p=1.0,
    )

    output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return_target = re.search(r"target:\s*(\d+)", output)
    return_target = return_target.group(1).strip() if return_target else ""
    print('------------------')
    print(f"model로부터 나오는 출력: {output}")
    print('----original----\n', text)
    print('----origianl target----\n', target)
    print('----new target----\n', return_target)
    print('------------------')

    return return_target

# Process each row in the data with tqdm progress bar
# 기존 데이터프레임에서 컬럼 이름만 추출 
columns = data[['ID', 'text', 'target']].columns 
# 빈 데이터프레임 생성 
aug_df = pd.DataFrame(columns=columns)

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
    i_d = row["ID"] + '_with_LLM_Trim_target'
    text = row["text"]
    label = row["target"]
    trimed_target = clean_text_with_model(text, label)

    new_row = pd.DataFrame({"ID": [i_d], "text": [text], "target": [label], "trimed_target": [trimed_target]})
    aug_df = pd.concat([aug_df, new_row], ignore_index=True)


augmentation_with_llm = pd.DataFrame(aug_df)
augmentation_with_llm.to_csv(output_file, index=False)

print(f"aug data saved to {output_file}")
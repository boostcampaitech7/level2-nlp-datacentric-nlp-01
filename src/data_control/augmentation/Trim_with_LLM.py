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
input_file = "/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/src/data_part_2.csv"
output_file = "Trim_AUG_data_with_LLM_Trimed_2.csv"
data = pd.read_csv(input_file)


def clean_text_with_model(text):
    messages = [
    {
    "role": "system",
    "content": ("""
    You are an expert Korean text augmentation system. Transform the given text according to these guidelines while preserving the original meaning:

    1. Text Transformation Techniques:
    - Replace words with synonyms or similar expressions
    - Modify particles (e.g., 은/는, 이/가, 을/를)
    - Add or modify adverbs for intensity or emotion
    - Alter word order
    - Convert between active and passive voice
    - Substitute Sino-Korean words with pure Korean equivalents

    2. Meaning Preservation:
    - Maintain the core message
    - Avoid factual distortions
    - Retain original tone, emotion, and nuance
    - Preserve specialized terms and proper nouns

    3. Formal Language Requirements:
    - Use formal written Korean
    - Employ official/formal sentence structures
    - Adhere to spelling and grammatical rules
    - Maintain a professional and objective tone
    - End sentences concisely with nouns, verb stems, or nominalizers, avoiding colloquial endings
    - IMPORTANT: Avoid unnecessarily extending sentences with formal expressions like '습니다' or '입니다'. Keep the text concise and to the point.

    4. Transformation Constraints:
    - Limit paraphrasing
    - Avoid adding unnecessary information
    - Keep sentence length within 1.5 times the original
    - Do not use abbreviations or contractions

    5. Quality Standards:
    - Ensure natural-sounding Korean sentences
    - Maintain logical consistency
    - Use clear, easily understandable expressions
    - Preserve professionalism and formality

    Chain-of-Thought Process:
    1. Analyze the original text for key elements (subject, predicate, objects, modifiers)
    2. Identify potential areas for transformation based on the techniques listed
    3. Apply transformations systematically, ensuring each change adheres to the guidelines
    4. Review the transformed text for meaning preservation and formal requirements
    5. Make final adjustments to meet quality standards and constraints

    Output your thought process for each step, then provide the final transformed text in this format:
    Transformed text: [Your augmented Korean text here]

    Remember: The goal is to create a diverse yet faithful augmentation that enhances the original text while maintaining its core meaning and formality. Crucially, avoid elongating sentences with formal endings like '습니다' or '입니다'; instead, aim for concise, professional expressions.

    Remember: Output format must always follow this prompt to maintain a few-shot style:
    text: """
    )},
    {"role": "user", "content": "text: '갤럭시노트8 주말 27만대 개통…시장은 불법 보조금 얼룩'"},
    {"role": "assistant", "content": "text: '주말 동안 갤럭시 노트 8, 27만 대 개통…불법 보조금으로 인해 시장에 먹칠'"},
    {"role": "user", "content": "text: 'NH투자 1월 옵션 만기일 매도 우세'"},
    {"role": "assistant", "content": "text: 'NH Investment & Securities 1월 옵션 만기일 매도 우세'"},
    {"role": "user", "content": "text: '공사업체 협박에 분쟁해결 명목 돈 받은 언론인 집행유예'"},
    {"role": "assistant", "content": "text: '건설 회사의 협박에 굴복하여 분쟁 조정 명분으로 돈을 받은 기자에게 집행유예가 선고'"},
    {"role": "user", "content": "text: '생명인증 앱 사용하면 대기가 끝'"},
    {"role": "assistant", "content": "text: '생체 인증 앱 활용 시 대기 시간이 단축'"},
    {"role": "user", "content": "text: 'LG G8씽큐 15일부터 예판…디스플레이 1회 무상교체 혜택'"},
    {"role": "assistant", "content": "text: 'LG G8 씽큐, 15일부터 사전 예약 시작…화면, 1회 무상 교체 혜택 제공'"},
    {"role": "user", "content": "text: '인공지능의 발전, 무엇을 향해 나아가는가'"},
    {"role": "assistant", "content": "text: '인공지능 기술 발전, 미래는 어디로?'"},
    {"role": "user", "content": "text: '공룡은 파충류보다는 타조·매 등 새에 가깝다'"},
    {"role": "assistant", "content": "text: '진화적으로 공룡은 파충류보다는 타조, 매와 같은 조류에 더 가까운 존재.'"},
    {"role": "user", "content": f"text: '{text}'"}]

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
        max_new_tokens=100,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print('----original----\n', text)
    print('----output----\n', output)
    aug_text = re.search(r"text:\s*'(.+)'", output)
    aug_text = aug_text.group(1).strip() if aug_text else ""
    print('----catched_text----\n', aug_text)

    return aug_text

# Process each row in the data with tqdm progress bar
aug_df = data[['ID', 'text', 'target']]

for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
    i_d = row["ID"] + '_LLM_Trim'
    text = row["text"]
    label = row["target"]

    aug_or_trimed_text = clean_text_with_model(text)

    new_row = pd.DataFrame({"ID": [i_d], "text": [aug_or_trimed_text], "target": [label]})
    aug_df = pd.concat([aug_df, new_row], ignore_index=True)


augmentation_with_llm = pd.DataFrame(aug_df)
augmentation_with_llm.to_csv(output_file, index=False)

print(f"aug data saved to {output_file}")
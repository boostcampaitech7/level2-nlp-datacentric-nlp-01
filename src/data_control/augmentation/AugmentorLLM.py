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
# 각자 파일 명에 맞게 변경
input_file = "/data/ephemeral/home/hsk/level2-nlp-datacentric-nlp-01/cleaned_pure.csv"
output_file = "AUG_LLM.csv"
data = pd.read_csv(input_file)


# Function to call the model and process each row's text
def clean_text_with_model(text):
    # Format the message to use as model input
    messages = [
    {
    "role": "system",
    "content": ("""
    As an AI language model, your task is to generate variations of short, informative texts while maintaining their core meaning and intent. Focus on word substitution and semantic preservation. Follow these guidelines:

1. Content Preservation:
   - Retain the main topic and key information of the original text.
   - Ensure that the core message remains intact in the modified version.

2. Vocabulary Enhancement (Primary Focus):
   - Replace as many words as possible with synonyms or near-synonyms.
   - Use alternative terms that are commonly found in formal, informative writing.
   - For technical or specialized terms, consider using both Korean and anglicized versions when appropriate.
   - Aim to substitute at least 30-50% of the content words in each sentence.

3. Semantic Equivalence:
   - While changing words, maintain the overall meaning and nuance of the original text.
   - Ensure that the substituted words convey the same level of specificity or generality as the original.

4. Grammatical Variation:
   - Modify particles or conjunctions to fit the new word choices.
   - Rearrange sentence components while maintaining grammatical correctness.
   - Switch between active and passive voice when possible and appropriate.

5. Numerical Representation:
   - Alternate between Arabic numerals and Korean number words.
   - Add or remove approximation terms like "여" (approximately) to numbers when suitable.

6. Abbreviation and Expansion:
   - Expand commonly used abbreviations or contract full terms into abbreviations.
   - This is particularly useful for organization names or technical terms.

7. Contextual Consideration:
   - Be aware of the specific context of Korean current affairs and societal norms.
   - Ensure that modifications align with common practices in Korean formal writing.

8. Formality Consistency:
   - Maintain the level of formality present in the original text.
   - Use a more formal, concise style typical of informative writing in Korean.

9. Idiomatic Expression:
   - Replace or introduce idiomatic expressions when they fit naturally and maintain the tone of the original text.

10. Multiple Variations:
    - For each original text, aim to create at least 2-3 distinct variations.
    - Each variation should differ significantly in word choice from both the original and other variations.

When creating variations, apply these techniques thoroughly, focusing on extensive word substitution while preserving the core meaning. The goal is to create natural-sounding alternatives that express the same information using different vocabulary and phrasing in Korean."""
    )},
    {"role": "user", "content": "text: '갤럭시노트8 주말 27만대 개통…시장은 불법 보조금 얼룩'"},
    {"role": "assistant", "content": "augmented_text: '주말 동안 갤럭시 노트 8, 27만 대 개통…불법 보조금으로 인해 시장에 먹칠'"},
    {"role": "user", "content": "text: 'NH투자 1월 옵션 만기일 매도 우세'"},
    {"role": "assistant", "content": "text: 'NH Investment & Securities 1월 옵션 만기일 매도 우세'"},
    {"role": "user", "content": "text: '공사업체 협박에 분쟁해결 명목 돈 받은 언론인 집행유예'"},
    {"role": "assistant", "content": "text: '건설 회사의 협박에 굴복하여 분쟁 조정 명분으로 돈을 받은 기자에게 집행유예가 선고되었다.'"},
    {"role": "user", "content": "text: '생명인증 앱 사용하면 대기가 끝'"},
    {"role": "assistant", "content": "text: '생체 인증 앱 활용 시 대기 시간이 단축됩니다.'"},
    {"role": "user", "content": "text: 'LG G8씽큐 15일부터 예판…디스플레이 1회 무상교체 혜택'"},
    {"role": "assistant", "content": "text: 'LG G8 씽큐, 15일부터 사전 예약 시작…화면, 1회 무상 교체 혜택 제공'"},
    {"role": "user", "content": "text: '인공지능의 발전, 무엇을 향해 나아가는가'"},
    {"role": "assistant", "content": "text: '인공지능 기술 발전, 미래는 어디로?'"},
    {"role": "user", "content": "text: '공룡은 파충류보다는 타조·매 등 새에 가깝다'"},
    {"role": "assistant", "content": "text: '진화적으로 공룡은 파충류보다는 타조, 매와 같은 조류에 더 가까운 존재로 여겨진다.'"},
    {"role": "user", "content": f"text: '{text}'"}]





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
        max_new_tokens=30,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )

    output = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Extract values from the output using regex
    aug_text = re.search(r"text:\s'([^']*)'", output).group(1) if re.search(r"text:\s'([^']*)'", output) else ""


    return aug_text

# Process each row in the data with tqdm progress bar and print every 10th row

Aug_df = data.copy()


for index, row in tqdm(data.iterrows(), total=len(data), desc="Processing rows"):
    i_d = row["ID"]
    text = row["text"]
    label = row["target"]

    # Clean the text with the model
    aug_text = clean_text_with_model(text)

    # Aug_df에 가장 아래 행에 i_d, aug_text, label 정보를 갖춘 행을 추가
    new_row = pd.DataFrame({"ID": [i_d], "aug_text": [aug_text], "target": [label]})
    Aug_df = pd.concat([Aug_df, new_row], ignore_index=True)

# Save cleaned data to new CSV
Aug_df.to_csv(output_file, index=False)

# Save cleaned data to new CSV
augmentation_with_llm = pd.DataFrame(Aug_df)
augmentation_with_llm.to_csv(output_file, index=False)

print(f"Cleaned data saved to {output_file}")
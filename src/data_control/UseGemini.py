from .noise_converter import NCGoogleGenAI
import pandas as pd

def routine_google_gen_ai():
    nc = NCGoogleGenAI(
        system_prompt="""As an AI language model, your task is to generate variations of short, informative texts while maintaining their core meaning and intent. Focus on word substitution and semantic preservation. Follow these guidelines:
        
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
        
        When creating variations, apply these techniques thoroughly, focusing on extensive word substitution while preserving the core meaning. The goal is to create natural-sounding alternatives that express the same information using different vocabulary and phrasing in Korean.
        
        Please answer the following question in JSON format, specifically in the structure: 
        {{
            "Answer": "your-answer-here"
        }}
        
        now do you task!
        """
    )
    
    texts = ["갤럭시노트8 주말 27만대 개통…시장은 불법 보조금 얼룩", 
             "NH투자 1월 옵션 만기일 매도 우세", 
             "공사업체 협박에 분쟁해결 명목 돈 받은 언론인 집행유예",
             "생명인증 앱 사용하면 대기가 끝",
             "LG G8씽큐 15일부터 예판…디스플레이 1회 무상교체 혜택",
             "인공지능의 발전, 무엇을 향해 나아가는가",
             "공룡은 파충류보다는 타조·매 등 새에 가깝다"]
    
    df = pd.DataFrame(texts, columns=['text'])
    
    df = nc.convert(df)
    df.to_csv('data/converted.csv', index=False)
    

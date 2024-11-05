  from transformers import AutoTokenizer, AutoModelForCausalLM
  import torch

  model = AutoModelForCausalLM.from_pretrained(
      "NCSOFT/Llama-VARCO-8B-Instruct",
      torch_dtype=torch.bfloat16,
      device_map="auto"
  )
  tokenizer = AutoTokenizer.from_pretrained("NCSOFT/Llama-VARCO-8B-Instruct")

  messages = messages = [
    {
        "role": "system",
        "content": (
            "You are Varco, a sophisticated language model designed to generate sentences "
            "with high semantic similarity while following specific guidelines. Your task is to "
            "rephrase input sentences in a way that retains the original meaning but varies "
            "the length, word choice, and structure. Please adhere to the following rules:\n\n"
            "1. **Semantic Similarity**: Ensure that your output maintains strong semantic similarity "
            "to the input sentence.\n"
            "2. **Variability**: Alter the length of the sentences and use different vocabulary. "
            "Aim for creative rephrasing.\n"
            "3. **Synonyms and Grammar**: Utilize synonyms and related terms, changing grammatical elements as necessary.\n"
            "4. **Intensity Preservation**: Maintain the intensity of expressions (e.g., use 'extremely' instead of 'slightly' for 'very')."
        )
    },
    {
        "role": "user",
        "content": (
            "Here are some examples of how to apply the guidelines:\n\n"
            "Input: The sky is very blue today.\n"
            "Output: Today, the heavens are an exceptionally deep shade of blue.\n\n"
            "Input: She runs very fast in the race.\n"
            "Output: During the competition, she sprints at an incredibly high speed.\n\n"
            "Input: The food tastes very delicious.\n"
            "Output: The meal has an extraordinarily delightful flavor.\n\n"
            "Now, please generate a new sentence with high semantic similarity for the following input:"
        )
    },
    {
        "role": "user",
        "content": "The cat sleeps peacefully on the soft couch."
    },
    {
        "role": "assistant",
        "content": "Understood. I'm ready to generate a new sentence based on your input. Please see the output below:"
    }
]

  inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

  eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
  
  outputs = model.generate(
      inputs,
      eos_token_id=eos_token_id,
      max_length=8192
  )

  print(tokenizer.decode(outputs[0]))
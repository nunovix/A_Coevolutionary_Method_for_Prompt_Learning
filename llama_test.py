import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from transformers import pipeline


def prompt_lm(system: str,
              user: str,
              assistant: str,
              model_id = "meta-llama/Llama-3.2-3B-Instruct",):
    
    print(model_id)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=256,
        num_beams = 3
    )

    output = outputs[0]["generated_text"][-1]
    print(output)
    print('\n\n')

    return output

a = "You give complete answers to questions."
b = "If I have 5 apples and I give away 4 do I still have at least 1 apple."
c = "Answer:"

prompt_lm(system=a, user=b, assistant=c)
#prompt_lm(system=a, user=b, assistant=c, model_id="microsoft/Phi-3-mini-4k-instruct")


"""
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map="cuda",)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map="cuda",)


import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You only answer with YES or NO."},
    {"role": "user", "content": "If I have 5 apples and I give away 4 do I still have at least 1 apple."},
    {"role": "assistant", "content": "Answer:"},
]
outputs = pipe(
    messages,
    max_new_tokens=256
)
print(outputs[0]["generated_text"][-1])
print('\n\n\n')
"""
"""
####################################################################################################



sentence = "You are a pirate chatbot who always responds in pirate speak!\n\n Who are you?"
input_ids = tokenizer(sentence, return_tensors="pt", return_attention_mask=True).to('cuda')

outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=256, num_beams = 3)

print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

####################################################################################################

"""

from evo_functions import convert_text_mistral_llama_3

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)

sentence = "[INST]You give complete answers to questions.\n\nIf I have 5 apples and I give away 4 do I still have at least 1 apple.[/INST]\n\nAnswer:"

sentence = convert_text_mistral_llama_3(sentence)

encoded_inputs = tokenizer(sentence, return_tensors="pt", return_attention_mask=True).to('cuda')

#prompt = tokenizer.encode(text_w_reflection, return_tensors="pt").to('cuda')
prompt_length = encoded_inputs['input_ids'][0].shape[0]
#print(f"PROMPT_LEN-->{prompt_length}")

output = model.generate(encoded_inputs['input_ids'], 
                            attention_mask=encoded_inputs['attention_mask'],
                            #past_key_values=cached_outputs.past_key_values, 
                            max_new_tokens=256, num_beams = 3)

print(f"\n\n\n\nInference-->{tokenizer.decode(output[0], skip_special_tokens=False)}")

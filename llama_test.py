import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, AutoModelForCausalLM

#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", device_map="cuda",)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map="cuda",)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", device_map="cuda",)


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
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256
)
print(outputs[0]["generated_text"][-1])
print('\n\n\n')

####################################################################################################



sentence = "You are a pirate chatbot who always responds in pirate speak!\n\n Who are you?"
input_ids = tokenizer(sentence, return_tensors="pt", return_attention_mask=True).to('cuda')

outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=256, num_beams = 3)

print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

####################################################################################################

from evo_functions import convert_text_mistral_llama_3

sentence = "[INST]You are a pirate chatbot who always responds in pirate speak!\n\n Who are you?\n\n[/INST] You do not know who I am"

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








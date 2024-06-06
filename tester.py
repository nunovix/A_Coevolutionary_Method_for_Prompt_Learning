import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["TORCH_CUDNN_DETERMINISTIC"] = "1"

import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from evo_functions import convert_text_mistral_phi3

config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

torch.random.manual_seed(0)

try:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
        quantization_config = config,
        attn_implementation="flash_attention_2",
        #attn_implementation='eager',
    )
except:
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
        quantization_config = config,
        #attn_implementation="flash_attention_2",
        attn_implementation='eager',
    )


tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

text = '[INST]Give me your top 10 of basketball players all time. Be sure to not include any made up names[/INST] Answer:'
messages = convert_text_mistral_phi3(text)
print(f"messages-->{messages}") 

model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
with torch.cuda.amp.autocast(), torch.inference_mode():
    output = model.generate(model_inputs, 
                                max_new_tokens=1000, 
                                do_sample=True)

decoded_output = tokenizer.batch_decode(output,
                                       skip_special_tokens=True)
print(f"\n\n\n")
print(decoded_output[0])
print(f"\n\n\n")























"""#import evo_functions as evo
from evo_functions import extract_lines_to_dict, extract_ContractNLI_data, prompt_creation_contractnli_span
import os
from collections import Counter

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

folder_path = 'ContractNLI_initial_population_prompts'
initial_population_prompts = extract_lines_to_dict(folder_path)
#evolutionary_prompts = evo.extract_lines_to_dict("evolutionary_prompts")
# evaluate test file from semeval or another task
data_expanded = extract_ContractNLI_data(type = 'dev')# evaluate test file from semeval or another task

print(data_expanded[0].keys())

text = data_expanded[0]['text']
statement = data_expanded[0]['statement']
label = data_expanded[0]['label']
spans = data_expanded[0]['spans']
spans_index = data_expanded[0]['spans_index']

for i in spans_index:
    #print(f"spans[{i}]-->{spans[i]}")
    #print(f"text[spans([{i}][0]:spans[{i}][1]]-->{text[spans[i][0]:spans[i][1]]}")
    #print(f"statement-->{statement}")
    #print(f"label-->{label}")
    pass
#print(f"initial_population_prompts.keys()-->{initial_population_prompts.keys()}")

statement = initial_population_prompts['statement_description'][0]
doc =initial_population_prompts['doc_description'][0]
answer = initial_population_prompts['answer_description'][0]
task  = initial_population_prompts['task_description'][0]

#print(f"\nstatement-->{statement}\n")
#print(f"doc-->{doc}\n")
#print(f"answer-->{answer}\n")
#print(f"task-->{task}\n")

prompts = prompt_creation_contractnli_span(data_expanded, task_description=task, doc_description=doc, statement_description=statement, answer_description=answer)

labels = []
for prompt in prompts:
    #print(f"\n\n\n\n\n\n\ntext-->{prompt['text']}")
    labels.append(prompt['label'])

print(f"counter-->{Counter(labels)}")"""
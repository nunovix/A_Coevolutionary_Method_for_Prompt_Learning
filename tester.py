#extract_LEXSUM_data(used_retrieved_file = False)

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import torch 
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
from evo_functions import convert_text_mistral_phi3

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
    attn_implementation="flash_attention_2"
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 


sentence = """<s><|user|>Below will be a question to a test with two possible answers: yer or no. Output only one of them

Is Lisbon the capital of Portugal?

OPTIONS:
- yes
- no<|end|>
<|assistant|> 
"""

#sentence = convert_text_mistral_phi3(sentence)

encoded_inputs = tokenizer(sentence,return_tensors="pt", return_attention_mask=True, padding=True).to('cuda')
input_len = encoded_inputs['input_ids'][0].shape[0]

with torch.inference_mode():
    output = model.generate(encoded_inputs['input_ids'],
                            attention_mask=encoded_inputs['attention_mask'], 
                            max_new_tokens=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            )
    

# Decode the generated sequence (excluding input)
generated_ids = output.sequences[0, encoded_inputs['input_ids'].shape[-1]:]  # Exclude the input part
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)


print(f"output.keys()-->{output.keys()}")
logits = output.scores[-1]  # logits of the last token

# Calculate probabilities
probabilities = F.softmax(logits, dim=-1)

yes_token_id = tokenizer.convert_tokens_to_ids("yes")
no_token_id = tokenizer.convert_tokens_to_ids("no")


# Check if the token id is valid and present
if yes_token_id != tokenizer.pad_token_id and yes_token_id < len(probabilities[0]):
    yes_token_prob = probabilities[0, yes_token_id].item()
else:
    yes_token_prob = 0.0

# Check if the token id is valid and present
if no_token_id != tokenizer.pad_token_id and no_token_id < len(probabilities[0]):
    no_token_prob = probabilities[0, no_token_id].item()
else:
    no_token_prob = 0.0


print(f"Generated text: {generated_text}")
print(f"Probability of 'yes': {yes_token_prob:.8f}")
print(f"Probability of 'no': {no_token_prob:.8f}")


def data_quality_assessment_save(task):
    # extract data acording to selected task
    # combine the validation and the training data
    # expect same format for all TRUE
    # create prompt part between ### and ###, one for each task
        # semeval
        # contract nli
        # mediqa sum
        # lex sum
    # append to common part
    # loop through and performance the inferences
        # in loop save the score
    # save new variation with the scores (in the same format they were extracted from but in reverse order of the scores)
    # so that in the implementation of extract from scored variation we just select the top k and o ordering is done there


    return None

from evo_functions import extract_SemEval_data


def data_quality_assessment_save(task: str):

    base_string = """The previous paragraph demarcated within ### and ### is a datapoint from a dataset. Your task is to determine whether this datapoint is a well-structured and informative example that would be valuable for evaluating the performance of a large-language model. An informative datapoint should be well-formatted, contain some usable knowledge of the task, and serve as a strong representation of the overall dataset.

OPTIONS: 
- yes
- no """
    
    # Step 1: Determine and execute the appropriate data extraction function based on `task`
    if task == "SemEval":
        data = extract_SemEval_data()  # Use your pre-existing function
    elif task == "task2":
        pass
    elif task == "task3":
        pass
    elif task == "task4":
        pass
    else:
        raise ValueError("Invalid task provided.")
    
    # Step 2: Generate a specific string for each data point
    processed_data = []
    for item in data:
        if task == "SemEval":
            generated_string = generate_string_for_task1(item)  # Custom function needed
        elif task == "task2":
            generated_string = generate_string_for_task2(item)
        elif task == "task3":
            generated_string = generate_string_for_task3(item)
        elif task == "task4":
            generated_string = generate_string_for_task4(item)
        
        # Combine with base_string
        combined_string = f"<s><|user|>\n###\n{generated_string}###\n\n {base_string}<|end|>\n<|assistant|>"
        
        # Add to processed_data
        processed_data.append({"original_data": item, "combined_string": combined_string, "score": None})
    
    # Step 3: Perform inferences in a loop
    for data_point in processed_data:
        inference_result = llm_inference_function(data_point["combined_string"])
        data_point["score"] = inference_result
    
    # Step 4: Sort the processed data by the `score` in descending order
    sorted_data = sorted(processed_data, key=lambda x: x["score"], reverse=True)
    
    return sorted_data


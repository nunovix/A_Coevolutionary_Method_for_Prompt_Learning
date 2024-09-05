# Functions to infer data quality and save results to new file. 
# Approach based on the method Ask-LLM from the paper https://arxiv.org/pdf/2402.09668v1

import torch
import torch.nn.functional as F
from evo_functions import load_model, extract_SemEval_data, extract_ContractNLI_data, extract_MEDIQASUM_data, extract_LEXSUM_data
import json
from tqdm import tqdm

def generate_string_for_semeval_data_quality(datapoint: dict):

    #task_description = "Natural Language Inference task between Clinical Trial Reports' (CTRs) Sections and a Statement made by a specialist. The Statement can be made about a single CTR or two CTRs. There are two possible relations between the statement and the CTRs: Entailment or Contradiction"
    primary_evidence = "\n".join(datapoint['primary_evidence'])
    statement = datapoint['statement']
    label = datapoint['label']

    semeval_text = f"""Primary Clinical Trial Report section:\n{primary_evidence}\n\n"""

    if 'secondary_evidence' in datapoint.keys():
        secondary_evidence = "\n".join(datapoint['secondary_evidence'])
        semeval_text = f"""{semeval_text}Secondary Clinical Trial Report section:\n{secondary_evidence}\n\n"""

    semeval_text = f"""{semeval_text}Statement: {statement}\n\nLabel: {label}"""

    return semeval_text

def generate_string_for_contractnli_data_quality(datapoint: dict):

    contractnli_text = f"""Non Disclosure Agreement:\n{datapoint['text']}\n\nStatement: {datapoint['statement']}\n\nLabel: {datapoint['label']}"""

    return contractnli_text

def generate_string_for_mediqasum_data_quality(datapoint: dict):
    print(datapoint.keys())

    mediqasum_text = f"""Patient-Doctor Dialogue:\n{datapoint['dialogue']}\n\nClinical Note:\n{datapoint['note']}"""

    return mediqasum_text

def generate_string_for_lexsum_data_quality(datapoint: dict):
    print(datapoint.keys())

    mediqasum_text = f"""Legal Act:\n{datapoint['reference']}\n\nSummary:\n{datapoint['summary']}"""

    return mediqasum_text


def data_quality_inference(data_quality_prompt, model, tokenizer):

    print(f"{data_quality_prompt}")

    encoded_inputs = tokenizer(data_quality_prompt, return_tensors="pt", return_attention_mask=True, padding=True).to('cuda')
    input_len = encoded_inputs['input_ids'][0].shape[0]

    with torch.inference_mode():
        output = model.generate(encoded_inputs['input_ids'],
                                attention_mask=encoded_inputs['attention_mask'], 
                                max_new_tokens=1,
                                return_dict_in_generate=True,
                                output_scores=True,
                                )
    


    generated_ids = output.sequences[0, :]  # Exclude the input part
    full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"generated_text-->{full_generated_text}")


    # Decode the generated sequence (excluding input)
    generated_ids = output.sequences[0, encoded_inputs['input_ids'].shape[-1]:]  # Exclude the input part
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    #print(f"generated_text-->{generated_text}")

    logits = output.scores[-1]  # logits of the last token

    # Calculate probabilities
    probabilities = F.softmax(logits, dim=-1)

    # \u2581 used before words to mark that it is a new word and not an attachable subword ie
    yes_token_id = tokenizer.convert_tokens_to_ids("\u2581YES")

    # Check if the token id is valid and present
    if yes_token_id != tokenizer.pad_token_id and yes_token_id < len(probabilities[0]):
        yes_token_prob = probabilities[0, yes_token_id].item()
    else:
        yes_token_prob = 0.0

    #print(f"YES_token_prob-->{yes_token_prob}")

    # Get the top 10 tokens with highest probabilities
    top_k_probs, top_k_ids = torch.topk(probabilities, 10, dim=-1)

    # Convert token ids to actual tokens and print them with their probabilities
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.squeeze().tolist())
    for token, prob in zip(top_k_tokens, top_k_probs.squeeze().tolist()):
        print(f"Token: {token}, Probability: {prob:.4f}")
    print(f"\n\n")
        
    return yes_token_prob

# function to assess data quality on the dataset folders from 
def data_quality_assessment_and_save(task: str,
                                     save = True,
                                     phi_model = '4k'):

    # base_data_quality_prompt = f"""The previous paragraph demarcated within ### and ### is a datapoint from a dataset. Your task is to determine whether this datapoint is a well-structured and informative example that would be valuable for evaluating the performance of a large-language model. An informative datapoint should be well-formatted, contain some usable knowledge of the task, and serve as a strong representation of the overall dataset.\n\nOPTIONS:\n- yes\n- no """
    base_data_quality_prompt = f"""Considering that an informative datapoint should be well-formatted, contain usable knowledge, and serve as a strong representation of the overall dataset, assess whether this datapoint is a well-structured and informative example that would be valuable for assessing the performance of a large language model, when considering a task related to classifying instances from the dataset. Answer with either YES or NO."""

    # Step 1: Determine and execute the appropriate data extraction function based on `task`
    if task == "SemEval":
        train_data = extract_SemEval_data(type = 'train')
        validation_data = extract_SemEval_data(type = 'dev')

    elif task == "ContractNLI":
        train_data = extract_ContractNLI_data(type = 'train')
        validation_data = extract_ContractNLI_data(type = 'dev')

    elif task == "MEDIQASUM":
        train_data = extract_MEDIQASUM_data(type = 'train')
        validation_data = extract_MEDIQASUM_data(type = 'valid')

    elif task == "LEXSUM":
        train_data = extract_LEXSUM_data(type = 'train')
        validation_data = extract_LEXSUM_data(type = 'validation')

    else:
        raise ValueError("Invalid task provided.")
    
    for train_datapoint in train_data:
        train_datapoint['set'] = 'train' 

    for val_datapoint in validation_data:
        val_datapoint['set'] = 'validation'

    full_data = train_data + validation_data
    #full_data = validation_data[:20]
    
    # Step 2: Generate a specific string for each data point
    for i in range(len(full_data)):
        if task == "SemEval":
            datapoint_string = generate_string_for_semeval_data_quality(full_data[i])
        elif task == "ContractNLI":
            datapoint_string = generate_string_for_contractnli_data_quality(full_data[i])
        elif task == "MEDIQASUM":
            datapoint_string = generate_string_for_mediqasum_data_quality(full_data[i])
        elif task == "LEXSUM":
            datapoint_string = generate_string_for_lexsum_data_quality(full_data[i])

        # Combine with base_string
        # data_quality_prompt = f"<s><|user|>\n###\n{datapoint_string}\n###\n\n{base_data_quality_prompt}<|end|>\n<|assistant|>"
        data_quality_prompt = f"<s><|user|>\nThe following textual description corresponds to a datapoint from a dataset.\n\n{datapoint_string}\n\n{base_data_quality_prompt}<|end|>\n<|assistant|>"
        
        # Add to data_list
        full_data[i]['data_quality_prompt'] = data_quality_prompt
        #processed_data.append({"original_data": item, "combined_string": combined_string, "score": None})
    
    if phi_model == '4k':
        model, tokenizer = load_model(checkpoint = "microsoft/Phi-3-mini-4k-instruct", quantized = True)
    elif phi_model == '128k':
        model, tokenizer = load_model(checkpoint = "microsoft/Phi-3-mini-128k-instruct", quantized = True)

    for i in tqdm(range(len(full_data)), desc = 'Performing Data Quality Inference'):
        data_quality_score = data_quality_inference(data_quality_prompt = full_data[i]["data_quality_prompt"], 
                                                    model = model, 
                                                    tokenizer = tokenizer)
        full_data[i]["score"] = data_quality_score
    
    # Step 4: Sort the processed data by the `score` in descending order
    full_data.sort(key=lambda x: x['score'], reverse=True)

    filtered_data = [{k: v for k, v in item.items() if k != 'data_quality_prompt'} for item in full_data]

    # Save the data to a JSON file
    if save == True:
        file_name = 'DATASETS/DATA_QUALITY/' + f"{task}_data_quality.json"
        with open(file_name, 'w') as json_file:
            json.dump(filtered_data, json_file, indent=4)
        
        print(f"Data with data quality assessment saved to {file_name}!")
    else:
        print(filtered_data[0].keys())
        print(f"Data NOT saved!")

    return None
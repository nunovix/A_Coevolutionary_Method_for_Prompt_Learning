# Functions to infer data quality and save results to new file. 
# Approach based on the method Ask-LLM from the paper https://arxiv.org/pdf/2402.09668v1

import torch
import torch.nn.functional as F
from evo_functions import load_model, extract_SemEval_data, extract_ContractNLI_data, extract_MEDIQASUM_data, extract_LegalSumTOSDR_data, prepare_text4llama3_instruct
import json
from tqdm import tqdm
import numpy as np
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

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

def generate_string_for_legalsumtosdr_data_quality(datapoint: dict):
    #print(f"keys-->{datapoint.keys()}\n\n")

    legalsumtosdr = f"""Terms of Services section: \n{datapoint['original_text']}\n\nSummary:\n{datapoint['reference_summary']}"""

    return legalsumtosdr


def data_quality_inference(data_quality_prompt, model, tokenizer, focus_ans = "positive"):

    #print(f"{data_quality_prompt}")

    encoded_inputs = tokenizer(data_quality_prompt, return_tensors="pt", return_attention_mask=True).to('cuda')
    input_len = encoded_inputs['input_ids'][0].shape[0]

    with torch.inference_mode():
        output = model.generate(encoded_inputs['input_ids'],
                                attention_mask=encoded_inputs['attention_mask'], 
                                max_new_tokens=1,
                                return_dict_in_generate=True,
                                output_scores=True,
                                do_sample=True,  # Enable sampling
                                top_k=50,        # Optional: Consider only top 50 tokens
                                top_p=0.9  
                                )

    generated_ids = output.sequences[0, :]  # Exclude the input part
    full_generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"generated_text-->{full_generated_text}\n\n")
    #full_generated_text_skip = tokenizer.decode(generated_ids, skip_special_tokens=True)
    #print(f"generated_text_skip-->{full_generated_text_skip}")


    # Decode the generated sequence (excluding input)
    generated_ids = output.sequences[0, encoded_inputs['input_ids'].shape[-1]:]  # Exclude the input part
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    #print(f"generated_text-->{generated_text}")

    logits = output.scores[-1]  # logits of the last token

    dq_score = yes_no_comp_score_calculator(logits, tokenizer, focus_ans=focus_ans)
    # Calculate probabilities
    probabilities = F.softmax(logits, dim=-1)

    # \u2581 used before words to mark that it is a new word and not an attachable subword ie
    #yes_token_id = tokenizer.convert_tokens_to_ids("\u2581YES") # phi 3 mini
    yess = ['ĠYES', 'ĠYes', 'Ġyes']
    yes_token_ids = [tokenizer.convert_tokens_to_ids(y) for y in yess]
    yes_token_probs = [probabilities[0, yes_token].item() for yes_token in yes_token_ids]
    #print(f"yes_token_probs-->{yes_token_probs}\n\n")
    total_yes_prob = sum(yes_token_probs)

    """
    # \u2581 used before words to mark that it is a new word and not an attachable subword ie
    #yes_token_id = tokenizer.convert_tokens_to_ids("\u2581YES") # phi 3 mini
    yes_token_id = tokenizer.convert_tokens_to_ids("ĠYES") # llama

    # Check if the token id is valid and present
    if yes_token_id != tokenizer.pad_token_id and yes_token_id < len(probabilities[0]):
        yes_token_prob = probabilities[0, yes_token_id].item()
    else:
        yes_token_prob = 0.0
    """

    #print(f"YES_token_prob-->{yes_token_prob}")

    # top 10 tokens
    top_k_probs, top_k_ids = torch.topk(probabilities, 5, dim=-1)

    # convert to text
    top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids.squeeze().tolist())
    for token, prob in zip(top_k_tokens, top_k_probs.squeeze().tolist()):
        print(f"Token: {token}, Probability: {prob:.4f}")
    print(f"\n\n")

    top_k_logits, top_k_ids = torch.topk(logits, 10, dim=-1)

    #for token, logit in zip(top_k_tokens, top_k_logits.squeeze().tolist()):
        #print(f"Token: {token}, Logit: {logit:.4f}")
    #print(f"\n\n")

    return dq_score

def yes_no_comp_score_calculator(logits, 
                                 tokenizer, 
                                 focus_ans = "positive", # focus labels ("positive" or "negative")
                                 ):
    print(f"focusing on {focus_ans} answers")

    # Calculate probabilities
    probabilities = F.softmax(logits, dim=-1)

    # \u2581 used before words to mark that it is a new word and not an attachable subword ie
    #yes_token_id = tokenizer.convert_tokens_to_ids("\u2581YES") # phi 3 mini
    yess = ['ĠYES', 'ĠYes', 'Ġyes']
    yes_token_ids = [tokenizer.convert_tokens_to_ids(y) for y in yess]
    yes_tokens_logits = [logits[0, yes_token].item() for yes_token in yes_token_ids]

    noss = ['ĠNO', 'ĠNo', 'Ġno']
    nos_token_ids = [tokenizer.convert_tokens_to_ids(n) for n in noss]
    nos_tokens_logits = [logits[0, no_token].item() for no_token in nos_token_ids]

    if focus_ans == 'positive':
        dq_score = np.sum(np.exp(yes_tokens_logits)) / (np.sum(np.exp(yes_tokens_logits)) + np.sum(np.exp(nos_tokens_logits)))
    elif focus_ans == 'negative':
        dq_score = np.sum(np.exp(nos_tokens_logits)) / (np.sum(np.exp(yes_tokens_logits)) + np.sum(np.exp(nos_tokens_logits)))
    
    print(dq_score)
    return dq_score


# function to assess data quality on the dataset folders from 
def data_quality_assessment_and_save(task: str,
                                     focus_ans = "positive",
                                     save = True,
                                     model = 'meta-llama/Llama-3.2-3B-Instruct'):
    
    if focus_ans != "positive" and focus_ans != "negative":
        sys.exit("Invalid focus_ans provided. Please provide either 'positive' or 'negative'.")

    # base_data_quality_prompt = f"""The previous paragraph demarcated within ### and ### is a datapoint from a dataset. Your task is to determine whether this datapoint is a well-structured and informative example that would be valuable for evaluating the performance of a large-language model. An informative datapoint should be well-formatted, contain some usable knowledge of the task, and serve as a strong representation of the overall dataset.\n\nOPTIONS:\n- yes\n- no """
    # base_data_quality_prompt = f"""Considering that an informative datapoint should be well-formatted, contain usable knowledge, and serve as a strong representation of the overall dataset, assess whether this datapoint is a well-structured and informative example that would be valuable for assessing the performance of a large language model, when considering a task related to classifying instances from the dataset. Answer with either YES or NO."""
    # base_data_quality_prompt = f"""Considering that an informative instance should be challenging to classify, illustrating the particular problems and corner cases that may exist in a given dataset, assess whether this instance corresponds to an informative example that would be valuable for assessing the performance of a large language model, when considering a task related to classifying instances from the dataset. Answer with either YES or NO."""
    # base_data_quality_prompt = f"""Consider the task of determining whether this instance is informative. An informative instance should be challenging to classify, illustrating the particular problems and corner cases that may exist in a given dataset. Your goal is to assess whether this instance corresponds to an informative example that would be valuable for assessing the performance of a large language model. Answer with YES if you deem this instance informative and challenging, or with NO if you do not."""
    # base_data_quality_prompt = f"""Consider the task of determining whether this instance is informative. An informative instance should be challenging to classify, illustrating the particular problems and corner cases that may exist in a given dataset. Your goal is to assess whether this instance corresponds to an informative example that would be valuable for assessing the performance of a large language model. Answer with Yes if you deem this instance informative and challenging, or with No if you do not. Answer with Yes only if you are sure about your answer."""
    # base_data_quality_prompt = f"""Consider the task of determining whether or not the instance is informative, in what regards exemplifying the contents of the dataset. An informative instance should be challenging to classify, illustrating the particular problems and corner cases that may exist in the dataset. Your goal is to assess whether this instance corresponds to an informative example that would be valuable for assessing the performance of a large language model. Answer affirmatively if you deem the instance to be informative and challenging, or negatively otherwise."""
    # base_data_quality_prompt = f"""Consider the task of determining whether or not the instance is uninformative, in what regards exemplifying the contents of the dataset. An uninformative instance should be easy to classify, failing to illustrate the particular problems and corner cases that may exist in the dataset. Your goal is to assess whether this instance corresponds to an uninformative example that should be ignored when assessing the performance of a large language model. Answer yes if you deem the instance to be uninformative, or negatively otherwise."""
    # base_data_quality_prompt = f"""Consider the task of determining whether or not a data instance is uninformative, in what regards exemplifying the contents of a dataset."""

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

    elif task == "LegalSumTOSDR":
        train_data = extract_LegalSumTOSDR_data(type = 'train')
        validation_data = []

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
        elif task == "LegalSumTOSDR":
            datapoint_string = generate_string_for_legalsumtosdr_data_quality(full_data[i])

        # Combine with base_string
        # data_quality_prompt = f"<s><|user|>\n###\n{datapoint_string}\n###\n\n{base_data_quality_prompt}<|end|>\n<|assistant|>"
        # data_quality_prompt = f"<s><|user|>\nThe following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}\n\n{base_data_quality_prompt}<|end|>\n<|assistant|>"
        # data_quality_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nThe following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}\n\n{base_data_quality_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: "
        # data_quality_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_data_quality_prompt}<|eot_id|<|start_header_id|>user<|end_header_id|>\n\nThe following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:"
        # data_quality_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{base_data_quality_prompt}<|eot_id|<|start_header_id|>user<|end_header_id|>\n\nThe following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}\n\nNotice that an uninformative instance should be very easy to analyze and classify, failing to illustrate the particular challenges and the corner cases that may exist in the complete dataset to which it belongs. Its contents provides little or no useful information, likely failing to elicit a meaningful response from its analysis. Your goal is to assess whether the instance corresponds to an uninformative example that should be ignored, e.g. when assessing the performance of a large language model over the complete dataset. Taking into account the aforementioned goal, attend carefully to the contents of the instance.\n\n{datapoint_string}\n\nAnswer affirmatively if you deem the instance to be uninformative, or negatively otherwise.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:"
        #system_part = f"Assume the role of an automated system for the processing of domain-specific documentation, such as clinical or legal documents. The accuracy, robustness, consistency, and faithfulness of the reasoning performed by the system is critical in this context, and it is important to carefully consider the domain-specific terminology, to handle linguistic constructs such as temporal associations or negations, and to have robustness to different writing styles and vocabularies."
        #data_quality_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_part}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nThe following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}\n\nConsider the task of determining whether or not the instance is uninformative, in what regards exemplifying the contents of the dataset. An uninformative instance should be very easy to analyze and classify, failing to illustrate the particular challenges and the corner cases that may exist in the complete dataset. Its contents provides little or no useful information, likely failing to elicit a meaningful response from its analysis. Your goal is to assess whether the instance corresponds to an uninformative example that should be ignored, e.g. when assessing the performance of a large language model over the complete dataset. Taking into account the aforementioned goal, attend carefully to the contents of the instance.\n\n{datapoint_string}\n\nAnswer affirmatively if you deem the instance to be uninformative, or negatively otherwise.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:"
        
        user_text_prompt = f"The following textual description corresponds to a particular instance from a dataset.\n\n{datapoint_string}\n\nConsider the task of determining whether or not the instance is uninformative, in what regards exemplifying the contents of the dataset. An uninformative instance should be very easy to analyze and classify, failing to illustrate the particular challenges and the corner cases that may exist in the complete dataset. Its contents provides little or no useful information, likely failing to elicit a meaningful response from its analysis. Your goal is to assess whether the instance corresponds to an uninformative example that should be ignored, e.g. when assessing the performance of a large language model over the complete dataset. Taking into account the aforementioned goal, attend carefully to the contents of the instance.\n\n{datapoint_string}\n\nAnswer affirmatively if you deem the instance to be uninformative, or negatively otherwise."
        
        #processed_data.append({"original_data": item, "combined_string": combined_string, "score": None})
    
    """if model == 'phi_4k':
        model, tokenizer = load_model(checkpoint = "microsoft/Phi-3-mini-4k-instruct", quantized = True)
    elif model == 'phi_128k':
        model, tokenizer = load_model(checkpoint = "microsoft/Phi-3-mini-128k-instruct", quantized = True)"""

    if 'llama' in model:
        prompt = prepare_text4llama3_instruct(user_text = user_text_prompt)
        model, tokenizer = load_model(checkpoint = model, quantized = True)
    else:
        sys.exit("not an implemented model")
    
    # Add to data_list
        full_data[i]['data_quality_prompt'] = prompt
    
    for i in tqdm(range(len(full_data)), desc = 'Performing Data Quality Inference'):
        data_quality_score = data_quality_inference(data_quality_prompt = full_data[i]["data_quality_prompt"], 
                                                    model = model, 
                                                    tokenizer = tokenizer,
                                                    focus_ans=focus_ans)
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
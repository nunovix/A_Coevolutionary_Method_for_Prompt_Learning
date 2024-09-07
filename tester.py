import os
import json
import numpy as np
from tqdm import tqdm

def extract_LegalSumTOSDR_data(folder_name='DATASETS/LegalSumTOSDR_data', 
                               type = 'val', 
                               used_retrieved_file = True,
                               retrieve_similar_examples = True,
                               save_retrieved = True,
                               use_data_sorted_by_dq = False,
                               ):
    if use_data_sorted_by_dq == True:
        file_path = "DATASETS/DATA_QUALITY/LegalSumTOSDR_data_quality.json"
    else:
        file_path = os.path.join(folder_name, f"{type}_w_retrieved.json")

    if used_retrieved_file == True and os.path.exists(file_path):
        # Load from a JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)
        print(f"Used data with already retrieved examples from {file_path}")
        return data_list
    
    full_data_file_name = 'tosdr_annotated_v1.json'
    file_path = os.path.join(folder_name, full_data_file_name)
    with open(file_path, mode='r') as file:
        full_data = json.load(file)

    print(full_data)
    print(type(full_data))

    data_expanded = []
    for uid in full_data:
        data_point = {'original_text': full_data[uid]['original_text'], 'reference_summary': full_data[uid]['reference_summary'], 'uid': full_data[uid]['uid']}
        data_expanded.append(data_point)

    np.random.shuffle(data_expanded)
    split_percentage = 0.8  
    split_index = int(len(data_expanded) * split_percentage)

    test_data = data_expanded[:split_index]
    val_data = data_expanded[split_index:]

    if save_retrieved == True:
        save_path = os.path.join(folder_name, f"test_w_retrieved.json")
        with open(save_path, 'w') as file:
            json.dump(test_data, file)
        print(f"Examples with retreival svaed to {save_path}")

        save_path = os.path.join(folder_name, f"val_w_retrieved.json")
        with open(save_path, 'w') as file:
            json.dump(val_data, file)
        print(f"Examples with retreival svaed to {save_path}")


def prompt_preds_leglasumtosdr(data_expanded, 
                               task_description, 
                               doc_description, 
                               answer_description,
                               model, 
                               tokenizer,
                               ):
    labels = []
    preds = []

    print_once_flag = 0

    for sample in tqdm(data_expanded):

        sentence = task_description + '\n\n'
        doc = sample['original_text']
        sentence = f"""[INST]{sentence}\n\n{doc_description}\n\n"{doc[:]}"\n\n{answer_description}[/INST]\n\nSummary:"""

        print(sentence)

        # conversion necessary for phi3 model
        if 'Phi3' in model_name_global:
            sentence = convert_text_mistral_phi3(sentence)
            #print(f"messages prompts-->{sentence}\n\n\n\n\n\n\n\n")
            #print(f"len(sentence)-->{len(sentence)}")
            #print(f"messages prompts-->{sentence[:200]}\n\n\n\n\n\n\n\n")
        
        labels.append(sample["reference_summary"])

        encoded_inputs = tokenizer(sentence[:], return_tensors="pt", return_attention_mask=True).to('cuda')
        #print(f"len(prompt)-->{len(prompt)}")
            
        prompt_length = encoded_inputs['input_ids'][0].shape[0]
        print(f"prompt_length in tokens-->{prompt_length}")
        with torch.inference_mode():
            output = model.generate(encoded_inputs['input_ids'], 
                                    attention_mask=encoded_inputs['attention_mask'], 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=50,
                                    do_sample=True,
                                    num_beams = 3
                                    )
            
        # Decode only the newly generated tokens
        # Skip the input tokens by starting the slice at input_length
        new_tokens = output[0, prompt_length:]

        if print_once_flag == 0:
            print(f"INFERENCE LEX SUM-->{tokenizer.decode(output[0])}")
            print(f"sample['summary/short']-->{sample['summary']}")
            print_once_flag = 1

        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds.append(pred)

    return labels, preds 


val_data = extract_LegalSumTOSDR_data(type='val')
test_data = extract_LegalSumTOSDR_data(type='test')

print(val_data)

prompt_preds_leglasumtosdr(val_data)
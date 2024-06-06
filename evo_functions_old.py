# versão antes da population ser um dictionario
# ou seja a eval estava separada do resto etc
# tmb não tinh acontract com os spans

import os
import json
import torch
from tqdm import tqdm # loading bars
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import random
import numpy as np
from genre.trie import MarisaTrie # to condition decoder
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex

# extract txt files in folder_path to dict with all the subprompts for task, ctr, statement and answer description
def extract_lines_to_dict(folder_path):
    files_dict = {}
    # Iterate through each file in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .txt file
        if file_name.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Read lines from the file
            with open(file_path, 'r') as file:
                lines = file.readlines()
            # Remove the newline characters from each line
            lines = [line.strip() for line in lines]
            # Remove the '.txt' extension and use the name as a key in the dictionary
            files_dict[file_name[:-4]] = lines
    return files_dict


# function to extract SemEval data to a list of dictionaries with the 
# id's, 'statement', 'primary_evidence', 'label' and  'secondary_evidence' if it existss
# based on code from https://aclanthology.org/2023.semeval-1.137.pdf
def extract_SemEval_data(folder = 'SemEval_data', type = 'dev'):

    type += '.json'
    split = type
    data = json.load(open(f"{folder}/{split}"))
    files = os.listdir(folder + "/CT json/")
    files.remove(".DS_Store")

    files_data = {file[:-5]:json.load(open(f"{folder}/CT json/{file}")) for file in files}

    data_expanded = []
    for _id, value in data.items():
        temp = {}
        temp["id"] = _id
        p_nctid = value["Primary_id"]
        s_nctid = value.get("Secondary_id")
        section_id = value["Section_id"]
        statement = value["Statement"]
        primary_evidence = files_data[p_nctid][section_id]
        temp["statement"] = statement
        temp["primary_evidence"] = primary_evidence
        temp["label"] = value["Label"]
        
        if s_nctid is not None:
            secondary_evidence = files_data[s_nctid][section_id]
            temp["secondary_evidence"] = secondary_evidence
        
        data_expanded.append(temp)

    return data_expanded

# based on code from https://github.com/jonathanherzig/commonsenseqa/blob/master/esim/reader_csqa.py
# function to extract 
def extract_CSQA_data(file_path = 'CSQA_data', type='dev'):
    if type == 'dev':
        file_path += '/dev_rand_split.jsonl'
    elif type == 'train':
        file_path += '/train_rand_split.jsonl'

    label_dict = {'A':'A', 'B':'B', 'C':'C', 'D':'D', 'E':'E'}

    data_expanded = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f):
            #print(f"line-->{line}")
            line = line.strip("\n")
            line = json.loads(line)
            if not line:
                continue
            question = line['question']['stem']
            choice = [c['text'] for c in line['question']['choices']]
            label = label_dict[line['answerKey']] if 'answerKey' in line else None

            temp = {}
            temp['question'] = question
            temp['choice'] = choice
            temp['label'] = label
            data_expanded.append(temp)
    
    return data_expanded


# function to extract ContractNLI data to a list of dictionaries with the 
def extract_ContractNLI_data(folder = 'ContractNLI_data', type = 'dev'):

    type += '.json'
    split = type
    data = json.load(open(f"{folder}/{split}"))
    #print(f"data-->{data}")
    #print(f"data.keys()-->{data.keys()}")

    # dictionary to store the statements
    statements = {}
    for i in data['labels']:
        statements[i] = data['labels'][i]['hypothesis']

    #print(f"data['documents'][0].keys()-->{data['documents'][0].keys()}")
    #print(f"data['documents'][0]['text']-->{data['documents'][0]['text']}")
    #print(f"data['documents'][0]['annotation_sets'][0]['annotations']-->{data['documents'][0]['annotation_sets'][0]['annotations']}")
    
    data_expanded = []
    for doc in data['documents']:
        text = doc['text']
        #print(f"text-->{text}")

        #print(f"doc[0]['annotation_sets'][0]['annotations']-->{doc['annotation_sets'][0]['annotations']}")
        for stat_name in doc['annotation_sets'][0]['annotations']:
            
            stat = statements[stat_name]
            #print(f"stat-->{stat}")

            label = doc['annotation_sets'][0]['annotations'][stat_name]['choice']
            #print(f"label-->{label}")

            # add to data_expanded, (each text has several statements associated with it)
            temp = {}
            temp['text'] = text
            temp["statement"] = stat
            temp["label"] = label
            data_expanded.append(temp)

    return data_expanded


# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('Entailment' or 'Contradiction')
# based on code from https://aclanthology.org/2023.semeval-1.137.pdf
def prompt_creation_semeval(data_expanded, task_description, ctr_description, statement_description, answer_description):
    samples = []
    for sample in data_expanded:
        prompt = task_description + '\n\n' + ctr_description + '\n\n'
        primary_evidence = "\n".join(sample['primary_evidence'])
        sentence = f"{prompt}Primary Trial\n{primary_evidence}"
        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary_evidence = "\n".join(sample['secondary_evidence'])
            sentence = f"{sentence}\n\nSecondary Trial\n{secondary_evidence}"
        #input_text = get_input_text(sentence, sample['statement'])
        stat = "".join(sample['statement'])
        sentence = f"[INST]{sentence}\n\n{statement_description}\n\n{stat}\n\n{answer_description}[/INST]\n\nANSWER:"
        temp = {"text":sentence, "label":sample['label']}
        samples.append(temp)

    return samples

# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('A', 'B', 'C', 'D' or 'E')
def prompt_creation_csqa(data_expanded, task_description, answer_description):
    samples = []
    letters = ['A', 'B', 'C', 'D', 'E']
    for sample in data_expanded:
        prompt = task_description + '\n' 

        sentence = f"{prompt}\n{sample['question']}"

        option_list = ''
        for i, j in zip(sample['choice'], letters):
            option = f"{j} - {i}\n"
            #print(f"option-->{option}")
            option_list += option

        #print(f"option_list-->{option_list}")

        #answer_description = 'Please provide only the letter of the correct option.'

        sentence = f"[INST]{sentence}\n{option_list}\n{answer_description}[/INST]\n\nANSWER:"
        temp = {"text":sentence, "label":sample['label']}
        samples.append(temp)

    return samples


# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('Entailment' or 'Contradiction' or 'NotMentioned')
def prompt_creation_contractnli(data_expanded, task_description, doc_description, statement_description, answer_description):
    samples = []
    for sample in data_expanded:
        prompt = f"[INST]{task_description}\n\n{doc_description}\n\n{sample['text']}\n\n{statement_description}\n\n{sample['statement']}\n\n{answer_description}[/INST]\n\nANSWER: "
        #print(f"prompt-->{prompt}")
        temp = {"text":prompt, "label":sample['label']}
        samples.append(temp)

    return samples


# function to extract yes or no from the generated string
def extract_yes_no_after_answer(s):
    if 'Yes' in s or 'YES' in s:
      return('YES')
    elif 'No' in s or 'NO' in s:
      return('NO')
    else:
      return('Answer not found')
    

def load_quantized_model(model_name: str):
    """
    :param model_name: Name or path of the model to be loaded.
    :return: Loaded quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        #load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map = 'cuda'
    )
    return model
    
# load model and tokenizer from hugging face
def load_model(checkpoint = "mistralai/Mistral-7B-Instruct-v0.2",
               quantized = True):

    torch.cuda.empty_cache()

    # loading
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, device_map = 'cuda')

    if quantized == False:
        model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map = 'cuda')
    elif quantized == True:
        model = load_quantized_model(checkpoint)

    # Check if the tokenizer has a pad token
    if tokenizer.pad_token is None:
        # Set pad_token_id to eos_token_id if pad_token is not defined
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    #model = model.to('cuda')  # Move model to GPU

    return model, tokenizer

# function to mutate prompts with a given LLM
# takes a mutation prompt (asking to paraphrase) and a subprompt (to be mutated), outputs the NEW mutated subprompt
def mutate_prompt(prompt, mutation_prompt, model, tokenizer, temperature = 1.0, top_p=0.8):
    instruction = '[INST]' + mutation_prompt + "\nINSTRUCTION: " + prompt + '[/INST]' + "\n\nNEW INSTRUCTION: "
    #print(f"instruction-->{instruction}")

    # Tokenize input and generate attention mask
    prompt = tokenizer.encode(instruction, return_tensors="pt").to('cuda')
    prompt_length = prompt[0].shape[0]

    try:
        # to improve efficiency
        with torch.inference_mode():
            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_length=800, do_sample=True, temperature = temperature, top_p=top_p)
    except:
        output = ''

    new_tokens = output[0, prompt_length:]
    mutated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")

    return mutated


# function to combine prompts using an LLM
# takes a combination prompt (asking to join two instructions) and two subprompt (to be combined), outputs the NEW combined subprompt
def combine_prompts(prompt_1, prompt_2, combination_prompt, model, tokenizer, temperature = 1.0, top_p=0.8):
    instruction = '[INST]' + combination_prompt + "\nINSTRUCTION 1: " + prompt_1 + "\nINSTRUCTION 2: " + prompt_2 + '[/INST]' + "\n\nNEW INSTRUCTION: "

    # Tokenize input and generate attention mask
    prompt = tokenizer.encode(instruction, return_tensors="pt").to('cuda')
    prompt_length = prompt[0].shape[0]
    # Tokenize input and generate attention mask

    try:
        # to improve efficiency
        with torch.inference_mode():
            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_length=1600, do_sample=True, temperature = temperature, top_p=top_p)
    except:
        output = ''

    new_tokens = output[0, prompt_length:]
    combined = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return combined

# used to limit decoding options
# given the task a set of possible answers is selected, which are then tokenized and used
# to create the MarisaTrie object
def get_Marisa_Trie(task, tokenizer):
    if task == 'SemEval':
        # IR ALTERAR A FUNÇÃO QUE depois CONVERTE PARA AS OPTIONS REAIS
        possibilities = ["YES", "Yes", "yes", "Entailment", "NO", "No", "no", "Contradiction"]
    elif task == 'CSQA':
        possibilities = ['A', 'B', 'C', 'D', 'E']
    elif task == 'ContractNLI':
        possibilities = ["YES", "Yes", "yes", "Entailment", "NO", "No", "no", "Contradiction", 'Not mentioned', 'Not Mentioned', 'NOT MENTIONED']
    
    encoded_possibilities = []
    for pos in possibilities:
        encoded_possibilities.append(tokenizer.encode(pos) + [tokenizer.eos_token_id])
    
    #print(encoded_possibilities)

    class MyMarisaTrie(MarisaTrie):
        def __init__(self, data): super().__init__(data)
        def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])
    trie = MyMarisaTrie(encoded_possibilities)

    return trie

# function to generate predictions for the task for a given prompt
# outputs both the predictions and the true labels
def semeval_predictions(model, tokenizer, samples, trie):

    '''
    class MyMarisaTrie(MarisaTrie):
        def __init__(self, data): super().__init__(data)
        def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])


    # limit options for decoding
    trie = MyMarisaTrie([tokenizer.encode("NO.") + [tokenizer.eos_token_id],
                         tokenizer.encode("No.") + [tokenizer.eos_token_id],
                         tokenizer.encode("YES.") + [tokenizer.eos_token_id],
                         tokenizer.encode("Yes.") + [tokenizer.eos_token_id]])

    
    # get max no. of tokens needed
    token_counts = []
    for sample in samples:
        # Tokenize input and generate attention mask
        encoding = tokenizer(sample["text"])
        token_counts.append(len(encoding['input_ids']))

    max_token_count = max(token_counts) + 6
    '''

    labels = []
    preds = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc = f"Generating Predictions with LLM"):
            labels.append(sample["label"])
            # Tokenize input and generate attention mask
            prompt = tokenizer.encode(sample["text"], return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to('cuda')
            prompt_length = prompt[0].shape[0]

            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, 
                                    max_length=prompt_length+6, 
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
        
            #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")
            #print(f"sampl['label']-->{sample['label']}")


            # Decode only the newly generated tokens
            # Skip the input tokens by starting the slice at input_length
            new_tokens = output[0, prompt_length:]

            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            preds.append(pred)
    return labels, preds


# function to generate predictions for the task for a given prompt
# outputs both the predictions and the true labels
def csqa_predictions(model, tokenizer, samples, trie):
    
    '''
    class MyMarisaTrie(MarisaTrie):
        def __init__(self, data): super().__init__(data)
        def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])

    # limit options for decoding
    trie = MyMarisaTrie([tokenizer.encode("A") + [tokenizer.eos_token_id],
                         tokenizer.encode("B") + [tokenizer.eos_token_id],
                         tokenizer.encode("C") + [tokenizer.eos_token_id],
                         tokenizer.encode("D") + [tokenizer.eos_token_id],
                         tokenizer.encode("E") + [tokenizer.eos_token_id]])

    # get max no. of tokens needed
    token_counts = []
    for sample in samples:
        # Tokenize input and generate attention mask
        encoding = tokenizer(sample["text"])
        token_counts.append(len(encoding['input_ids']))

    max_token_count = max(token_counts) + 6
    '''

    labels = []
    preds = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc = f"Generating Predictions with LLM"):
            labels.append(sample["label"])
            # Tokenize input and generate attention mask
            prompt = tokenizer.encode(sample["text"], return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to('cuda')
            prompt_length = prompt[0].shape[0]

            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, 
                                    max_length=prompt_length+6, 
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
            
            #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")

            # Decode only the newly generated tokens
            # Skip the input tokens by starting the slice at input_length
            new_tokens = output[0, prompt_length:]

            #print(f"tokenizer.decode(new_tokens)-->{tokenizer.decode(new_tokens)}")
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            preds.append(pred)
            #print(f"preds-->{preds}")

    return labels, preds


# function to generate predictions for the task for a given prompt
# outputs both the predictions and the true labels
def contractnli_predictions(model, tokenizer, samples, trie):

    """
    class MyMarisaTrie(MarisaTrie):
        def __init__(self, data): super().__init__(data)
        def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])

    # limit options for decoding
    trie = MyMarisaTrie([tokenizer.encode("YES") + [tokenizer.eos_token_id],
                         tokenizer.encode("Yes") + [tokenizer.eos_token_id],
                         tokenizer.encode("yes") + [tokenizer.eos_token_id],
                         tokenizer.encode("Entailment") + [tokenizer.eos_token_id],
                         tokenizer.encode("NO") + [tokenizer.eos_token_id],
                         tokenizer.encode("No") + [tokenizer.eos_token_id],
                         tokenizer.encode("no") + [tokenizer.eos_token_id],
                         tokenizer.encode("Contradiction") + [tokenizer.eos_token_id],
                         tokenizer.encode("Not mentioned") + [tokenizer.eos_token_id],
                         tokenizer.encode("Not Mentioned") + [tokenizer.eos_token_id]])

    
    # get max no. of tokens needed
    token_counts = []
    for sample in samples:
        # Tokenize input and generate attention mask
        encoding = tokenizer(sample["text"])
        token_counts.append(len(encoding['input_ids']))

    max_token_count = max(token_counts) + 6
    """

    labels = []
    preds = []
    with torch.inference_mode():
        for sample in tqdm(samples, desc = f"Generating Predictions with LLM"):
            #torch.cuda.empty_cache()
            labels.append(sample["label"])
            # Tokenize input and generate attention mask
            prompt = tokenizer.encode(sample["text"], return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to('cuda')
            prompt_length = prompt[0].shape[0]
            #print(f"prompt_length-->{prompt_length}")
            try:
                output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, 
                                    max_length=prompt_length + 6, 
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
            except:
                print('prompt too long!')
                output = tokenizer.encode(sample["text"]+'Error', return_tensors="pt", padding=True, truncation=True, return_attention_mask=True).to('cuda')

            #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")
            #print(f"sample['label']-->{sample['label']}")

            # Decode only the newly generated tokens
            # Skip the input tokens by starting the slice at input_length
            new_tokens = output[0, prompt_length:]

            #print(f"tokenizer.decode(new_tokens)-->{tokenizer.decode(new_tokens)}")
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"pred-->{pred}")
            print(f"sample['label']-->{sample['label']}")
            preds.append(pred)
            #print(f"preds-->{preds}")

    return labels, preds


# takes prediction array and converts to entailment or contradiction labels
def convert_preds_from_yesno(preds):
    preds_2 = []
    no_of_not_founds = 0
    for i in preds:
        #print(f"i-->{i}")
        if i == 'YES' or i == 'Yes' or i == 'yes' or i == 'Entailment':
            preds_2.append('Entailment')
        elif i == 'NO' or i == 'No' or i == 'no' or i == 'Contradiction':
            preds_2.append('Contradiction')
        else:
            print('olha as labels')
            preds_2.append('Contradiction')
            no_of_not_founds += 1
    return preds_2, no_of_not_founds


def convert_preds_from_yesno_contractnli(preds):
    preds_2 = []
    no_of_not_founds = 0
    for i in preds:
        if i == 'YES' or i == 'Yes' or i == 'yes' or i == 'Entailment':
            preds_2.append('Entailment')
        elif i == 'NO' or i == 'No' or i == 'no' or i == 'Contradiction':
            preds_2.append('Contradiction')
        elif i == 'Not mentioned' or i == 'Not Mentioned' or i == 'NOT MENTIONED':
            preds_2.append('NotMentioned')
        else:
            print('olha as labels')
            preds_2.append('Contradiction')
            no_of_not_founds += 1
    return preds_2, no_of_not_founds


# function to evaluate prompt population
# outputs a list with the scores for each prompt
# n_samples is the no. of samples where the evaluation will be done
def eval_pop(n_pop, population_prompts, data_expanded, model, tokenizer, trie, n_samples=0, task='SemEval'):
    if task != 'SemEval' and task != 'CSQA' and task != 'ContractNLI':
        return None

    scores = []

    if n_samples == 0 or n_samples > len(data_expanded):
        n_samples = len(data_expanded)

    if task == "SemEval":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            samples = prompt_creation_semeval(data_expanded, 
                                                population_prompts['task_description'][i], 
                                                population_prompts['ctr_description'][i], 
                                                population_prompts['statement_description'][i], 
                                                population_prompts['answer_description'][i])

            labels, predictions = semeval_predictions(model, tokenizer, samples[:n_samples], trie)
            preds, n_not_founds = convert_preds_from_yesno(predictions)
            #print(f"predictions-->{predictions}")
            #print(f"preds-->{preds}")
            #print(f"labels-->{labels}")
            print(f"n_not_founds-->{n_not_founds}")
            score = f1_score(y_true=labels, y_pred=preds, pos_label="Entailment")
            scores.append(score)

    elif task == "CSQA":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            samples = prompt_creation_csqa(data_expanded, 
                                           population_prompts['task_description'][i],
                                           population_prompts['answer_description'][i])

            labels, predictions = csqa_predictions(model, tokenizer, samples[:n_samples], trie)
            score = accuracy_score(y_true=labels, y_pred=predictions)
            scores.append(score)

    elif task == "ContractNLI":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            samples = prompt_creation_contractnli(data_expanded, 
                                                population_prompts['task_description'][i], 
                                                population_prompts['doc_description'][i], 
                                                population_prompts['statement_description'][i], 
                                                population_prompts['answer_description'][i])

            labels, predictions = contractnli_predictions(model, tokenizer, samples[:n_samples], trie)
            preds, n_not_founds = convert_preds_from_yesno_contractnli(predictions)
            print(f"predictions-->{predictions}")
            print(f"preds-->{preds}")
            print(f"labels-->{labels}")
            print(f"n_not_founds-->{n_not_founds}")
            score = accuracy_score(y_true=labels, y_pred=preds)
            scores.append(score)

    return scores

# create folder to store each run of the evo_alg function
def create_root_folder(task):
    # Format: Runs_YYYY-MM-DD_HH-MM-SS
    folder_name = datetime.now().strftime(f"RUNS_{task}/Runs_%Y-%m-%d_%H-%M-%S")

    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# saving population at iteration
def save_population(iteration, population_dict, additional_list, root_folder, keep_list):
    # Create a folder for the current iteration
    iteration_folder = os.path.join(root_folder, f"Iteration_{iteration}")
    os.makedirs(iteration_folder, exist_ok=True)
    
    # Save each key in the population dictionary as a .txt file
    for key, values in population_dict.items():
        file_path = os.path.join(iteration_folder, f"{key}.txt")
        with open(file_path, 'w') as file:
            for value in values:
                file.write(f"{value}\n")
                file.write("----------\n")  # Optional separator line
    
    # Save the additional list in a separate .txt file
    additional_file_path = os.path.join(iteration_folder, "evaluations.txt")
    with open(additional_file_path, 'w') as file:
        for item in additional_list:
            file.write(f"{item}\n")
    
    # Save the additional list in a separate .txt file
    additional_file_path = os.path.join(iteration_folder, "keep_list.txt")
    with open(additional_file_path, 'w') as file:
        for item in keep_list:
            file.write(f"{item}\n")

    return None

def save_details(root_folder, n_pop,
                 n_top, 
                 start_time, 
                 end_time,
                 n_combinations,
                 patience,
                 max_iter,
                 iter,
                 temperature,
                 top_p,
                 best_score_iterations,
                 eval_data,
                 data_size,
                 task,
                 model_name,
                 quantize_model_4bits):
    
    dif = end_time - start_time
    avg_per_iter = dif.total_seconds()/iter
    avg_per_iter = timedelta(seconds = avg_per_iter)

    # Save the additional list in a separate .txt file
    additional_file_path = os.path.join(root_folder, "details.txt")
    with open(additional_file_path, 'w') as file:
        file.write(f"Task: {task}\n")
        file.write(f"Start time: {start_time}\n")
        file.write(f"End time: {end_time}\n")
        file.write(f"Total elapsed time: {dif}\n")
        file.write(f"No. of iterations: {iter}\n")
        file.write(f"Average time per iteration: {avg_per_iter}\n\n")

        file.write(f"Initial population size: {n_pop}\n")
        file.write(f"Population size that's kept for next iteration: {n_pop}\n")
        file.write(f"How many of the top performers are being kept (the rest are randomized): {n_top}\n")
        file.write(f"No. of mutations generated per iteration: {n_pop}\n")
        file.write(f"No. of combinations generated per iteration: {n_combinations}\n")

        file.write(f"Max no. of iterations allowed: {max_iter}\n")
        file.write(f"Patience: {patience}\n")
    
        file.write(f"Decoder temperature in mutation and combiantions: {temperature}\n")
        file.write(f"Top-p in sampling for mutation and combiantions: {top_p}\n\n")

        file.write(f"Evaluation done on: {eval_data} set\n")
        file.write(f"With {data_size} examples\n\n")

        file.write(f"Name of the model used: {model_name} \n")
        file.write(f"4 bit quantization: {quantize_model_4bits} \n")

    # Save best f1 score at each iteration
    additional_file_path = os.path.join(root_folder, "scores_evo.txt")
    with open(additional_file_path, 'w') as file:
        for item in best_score_iterations:
            file.write(f"{item}\n")

    return None


# function to sort prompt population based on evaluation values
def sort_pop(population, eval):

    sorted_indices = np.argsort(eval)[:][::-1]
    sorted_pop = {key: [value[i] for i in sorted_indices if i < len(value)] for key, value in population.items()}
    sorted_eval = [eval[i] for i in sorted_indices if i < len(eval)]

    return sorted_pop, sorted_eval

# function to select population and respectives evaluations in an exploratory or exploitanional way
# n_pop is the number of individuals we want to keep
# n_top is the number of best performing individuals we want to keep, while the rest will be randomized
# if n_pop=n_top then full greedy search is done (only top candidates are kept
def pop_selection(pop, eval, 
                  n_pop, # population size to be kept
                  n_top, # no. of top candidates to select (rest are randomized)
                  ):
    sorted_pop, sorted_eval = sort_pop(pop, eval)
    
    keep_list = list(range(n_top)) + random.sample(range(n_top, len(eval)), k=n_pop-n_top)
    keep_list.sort()
    print(f"keep_list-->{keep_list}")

    keep_pop = {key: [value[i] for i in keep_list if i < len(value)] for key, value in sorted_pop.items()}
    keep_eval = [sorted_eval[i] for i in keep_list]

    return keep_pop, keep_eval, keep_list

# function to run the evolutionary alg, with a initial population of prompts
# evolutionary prompts (1 for mutation, 1 for combination)
# hf model and tokenizer
# hyperparameters of the algorithm
def evo_alg(task, initial_population_prompts, evolutionary_prompts,
            model_name = "mistralai/Mistral-7B-Instruct-v0.2",
            quantize_model_4bits = True,
            n_pop = 5, # initial population size and the number of elements kepts at each iteration
            n_top = 5, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
            n_combinations = 10,
            patience = 10,
            max_iter = 50,
            temperature = 1.0, #temperature for decoding combined and mutated
            top_p = 0.8, #sampling for decoding combined and mutated
            save = True,
            eval_data = 'dev', # dev or train
            data_size = 0): # no. of samples where the prompts are evaluated, if =0 all are used
    
    if task != 'SemEval' and task != 'CSQA' and task != 'ContractNLI':
        print(f"Not right task selected")
        return None
    
    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized=quantize_model_4bits)

    trie = get_Marisa_Trie(task, tokenizer)
    
    # list to save best score at each iteration
    best_score_iterations = []

    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task)
        print(f"Root folder created: {root_folder}")

    if task == 'SemEval':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = eval_data)

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)
    
    patience_counter = 0
    iter = 0

    population = {key: [] for key in initial_population_prompts.keys()}
    initial_eval = eval_pop(n_pop = n_pop, population_prompts = initial_population_prompts, data_expanded = data_expanded, 
                            model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size, task=task)
    
    population = initial_population_prompts
    print(f"initial_eval-->{initial_eval}")
    best_score_iterations.append(max(initial_eval))
    
    if save == True:
        save_population('initial', population, initial_eval, root_folder, list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 
        mutated_population = {key: [] for key in initial_population_prompts.keys()}
        combined_population = {key: [] for key in initial_population_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop), desc = f"iteration {iter} - Mutating prompts"):
            # iterate through the subprompts
            for j in initial_population_prompts.keys():

                # mutate each subprompt and add to the mutated population
                mutated = mutate_prompt(initial_population_prompts[j][i], evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 

                mutated_population[j].append(mutated)
                population[j].append(mutated)
        
        mutated_eval = eval_pop(n_pop = n_pop, population_prompts = mutated_population, 
                                data_expanded = data_expanded, 
                                model=model, tokenizer=tokenizer, trie=trie,
                                n_samples = data_size, task=task)
        
        eval = initial_eval + mutated_eval
        #print(f"eval-->{eval}")

        for i in tqdm(range(n_combinations), desc = f"iteration {iter} - Combining prompts"):
            sel4comb = random.choices(range(n_pop + n_pop), weights=eval, k=2)
            # iterate through the subprompts
            for j in initial_population_prompts.keys():

                # combine each subprompt randomly selected and add to the combined and total population
                combined = combine_prompts(population[j][sel4comb[0]], population[j][sel4comb[1]], 
                                           evolutionary_prompts['combination_prompts'][0], model, tokenizer,
                                           temperature=temperature, top_p=top_p)
                combined_population[j].append(combined)
                population[j].append(combined)
        
        combined_eval = eval_pop(n_pop = n_combinations, population_prompts = combined_population, data_expanded=data_expanded, 
                                 model=model, tokenizer=tokenizer, trie=trie,
                                 n_samples = data_size, task=task)
        eval += combined_eval

         # if improved patience returns to 0
        if max(eval) > max(initial_eval):
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population, sorted_eval = sort_pop(population, eval)
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_eval}")

        # Create a new dictionary with the same keys, but values are lists with only the selected indices
        # how the population is being maintained
        keep_pop, keep_eval, keep_list = pop_selection(sorted_population, sorted_eval, n_pop, n_top)
        population = keep_pop
        initial_eval = keep_eval
        print(f"evaluation of keepers for next gen-->{initial_eval}")
        print(f"keep_list-->{keep_list}")

        # Call the function
        if save == True:
            save_population(iter+1, sorted_population, sorted_eval, root_folder, keep_list)
            best_score_iterations.append(max(eval))
        # increase iter counter
        iter += 1

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_prompt = {key: [value[i] for i in [0] if i < len(value)] for key, value in keep_pop.items()}
    best_eval = initial_eval[0]
    if save == True:
            save_population('best', best_prompt, [best_eval], root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            n_combinations,
                            patience,
                            max_iter,
                            iter,
                            temperature,
                            top_p,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits)
            create_plots_from_RUNS_folder(root_folder)

    return best_prompt, best_score_iterations

# keep in mind
# dá para otimizar lieiramente, porque o segundo sort dos elementos 
# não é preciso poqeure já fazes sort de tudo
##### PLOTTING

# function that creates and saves the plots for the iterations evolution
def plot_and_save_scores(all_scores, max_scores, directory_path, display_only_top_values, iteration_folders, 
                         y_min, y_max, score, keep_list):
    # Prepare the x-axis values (iteration numbers)
    x_values = range(len(all_scores))
    # Custom x-axis labels from the iteration folder names
    x_labels = [folder.replace('Iteration_', '') for folder in iteration_folders]

    # Create the plot
    plt.figure(figsize=(10, 6))

    if display_only_top_values:
        # Plot filename for only top scores
        plot_filename = 'top_scores_plot.png'
        # Plot only the maximum scores for each iteration
        plt.plot(x_values, max_scores, '-o', color='darkblue', label='Top Scores')
    
    else:
        # Plot filename for all scores
        plot_filename = 'all_scores_plot.png'
        # Define a list of base colors for the iterations
        base_colors = plt.cm.get_cmap('tab20', len(all_scores))

        # Plot all scores with lighter color for those not in keep_list
        for i, (scores, keep_indices) in enumerate(zip(all_scores, keep_list)):
            base_color = base_colors(i)  # Get the base color for this iteration
            # Convert base color to RGBA and then lighten the color for non-highlighted points
            lighter_color = to_rgba(base_color, alpha=0.12)  # Adjust alpha to make lighter
            # Plot all scores in lighter color
            plt.scatter([i] * len(scores), scores, color=lighter_color, label='Iteration {}'.format(i) if i == 0 else "")
            # Overlay highlighted scores in original color
            highlighted_scores = [scores[idx] for idx in keep_indices if idx < len(scores)]
            plt.scatter([i] * len(highlighted_scores), highlighted_scores, color=base_color)

        # Plot the line for top scores in a consistent color
        plt.plot(x_values, max_scores, '-o', color='black', label='Top Scores')

    # Labeling the plot
    plt.title('Scores by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel(score)
    plt.xticks(x_values, x_labels, rotation='vertical')  # Set custom x-axis labels
    plt.ylim(y_min, y_max)  # Set the y-axis range
    #plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(directory_path, plot_filename))
    plt.close()  # Close the plot to free up memory
    
    print("Plots have been saved to:", directory_path)

# function that takes file path to folder with runs, created during the evolution of the model
# also takes y axix scale
# and label for the y axis, which is the score
def create_plots_from_RUNS_folder(directory_path):
    if "SemEval" in directory_path:
        ymin = 0.60
        ymax = 0.80
        score = 'F1-Score'
    elif 'CSQA' in directory_path:
        ymin = 0.50
        ymax = 0.80
        score = 'Accuracy'
    elif 'ContractNLI' in directory_path:
        ymin = 0.20
        ymax = 0.80
        score = 'Accuracy'
    else:
        print(f"Incorrect task name")
        return None

    # List all items in the directory
    items = os.listdir(directory_path)

    # Filter out items that are not directories or are 'Iteration_best'
    iteration_folders = [item for item in items 
                        if os.path.isdir(os.path.join(directory_path, item)) and item != 'Iteration_best']

    # Custom sorting function
    def custom_sort(folder_name):
        if folder_name == 'Iteration_initial':
            return -1  # Ensure 'Iteration_initial' comes first
        else:
            # Extract the iteration number and convert it to an integer for proper numerical sorting
            num_part = folder_name.split('_')[-1]
            return int(num_part) if num_part.isdigit() else float('inf')  # Non-numeric suffixes go at the end

    # Sort the folders using the custom function
    iteration_folders.sort(key=custom_sort)

    # Initialize a list to hold all scores lists and a list for max scores
    all_scores = []
    max_scores = []
    keep_lists = []

    for folder in iteration_folders:
        # Construct the path to the evaluation.txt file
        file_path = os.path.join(directory_path, folder, 'evaluations.txt')
        
        # Initialize a list to hold scores for this iteration
        scores = []
        
        # Open the file and read the scores
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a float and append to the scores list
                scores.append(float(line.strip()))
        
        # Append this iteration's scores to the all_scores list
        all_scores.append(scores)
        
        # Find and append the max score for this iteration to the max_scores list
        max_scores.append(max(scores))

        file_path = os.path.join(directory_path, folder, 'keep_list.txt')
        keep_list = []
        # Open the file and read the scores
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a float and append to the scores list
                keep_list.append(int(line.strip()))

        keep_lists.append(keep_list)

    plot_and_save_scores(all_scores, max_scores, directory_path, False, iteration_folders, ymin, ymax, score, keep_lists)  # For all scores
    plot_and_save_scores(all_scores, max_scores, directory_path, True, iteration_folders, ymin, ymax, score, keep_lists)  # For only top scores

    return None
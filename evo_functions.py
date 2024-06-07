# NOTAS FUTURAS
# O CSQA está sem o phi 3 implementado porque ainda tem o prompt making separado do eval

import os
import sys
import json
import torch
from tqdm import tqdm # loading bars
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import random
import numpy as np
import csv
import re
from trie import MarisaTrie # to condition decoder
#from marisa_trie import Trie as MarisaTrie 
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
from copy import deepcopy

# embbedings library used in retrieval
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# fucntion to evaluate mediqa sum summarization (avg between BLEU, ROUGE and BERTSCORE)
import mediqasum_evaluation

# semeval evaluation
from semeval_evaluation import main as semeval_test_evaluation

# for batching
from torch.utils.data import DataLoader, TensorDataset

# function to select dataset and extract initial population of prompts 
# and the prompts to perform mutation and crossover
# also returns trie (to condition decoding for NLI tasks)
def sel_task_dataset_initial_prompts_evo_prompts(task_name,
                                                tokenizer,
                                                w_self_reasoning=False,
                                                w_one_shot=False,
                                                w_highlight=False,
                                                ):

    if task_name == 'SemEval':
        prompts_path = 'INITIAL_PROMPTS/SemEval_initial_population_prompts'
        data_expanded = extract_SemEval_data(extract_examples = w_one_shot)
        trie = get_Marisa_Trie(task_name, tokenizer)

    elif task_name == 'ContractNLI':
        prompts_path = 'INITIAL_PROMPTS/ContractNLI_initial_population_prompts'
        data_expanded = extract_ContractNLI_data()
        trie = get_Marisa_Trie(task_name, tokenizer)

    elif task_name == 'MEDIQASUM':
        prompts_path = 'INITIAL_PROMPTS/MEDIQASUM_initial_population_prompts'
        data_expanded = extract_MEDIQASUM_data(retrieve_similar_examples = w_one_shot)
        trie = None

    else:
        print(f"'{task_name}' is not a valid task name!")
        sys.exit()
    
    # take initial prompts from appropriate folder
    initial_population_prompts = extract_lines_to_dict(prompts_path, 
                                                       task = task_name, 
                                                       task_w_one_shot=w_one_shot,
                                                       task_w_self_reasoning=w_self_reasoning,
                                                       task_w_highlight = w_highlight
                                                       )
    
    # check if number of examples in each subpromtp is the same
    tam = []
    for key in initial_population_prompts:
        tam.append(len(initial_population_prompts[key]))
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"The no. of elements in each subprompt differs")
        sys.exit()

    # prompts to perform mutation and crossover
    evolutionary_prompts = extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "Evo_prompts")
    
    return data_expanded, initial_population_prompts, evolutionary_prompts, trie

# print memory usage from gpu
def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# function to convert string to be tokenized from the format expected by the mistral model to the 
# format expected by the phi 3 model
def convert_text_mistral_phi3(input_string):
    # Check if both start and end markers are present
    start_index = input_string.find('[INST]')
    end_index = input_string.find('[/INST]')
    
    if start_index == -1 and end_index == -1:
        print(f"ERROR: Required markers '[INST]' and '[/INST]' are not found in the input.")
        sys.exit()
    
    # Extract content between markers, adjusting for length of '[INST]'
    if start_index != -1 and end_index != -1:
        content = input_string[start_index + 6:end_index].strip()
    if start_index != -1 and end_index == -1:
        content = input_string[start_index + 6:].strip()
    
    # Create the message dictionary
    phi_format = f"<|user|>\n{content}<|end|>\n<|assistant|>\n{input_string[end_index+7:]}"
    message = [{"role": "user", "content": content},
                {"role": "assistant", "content":  input_string[end_index+7:]},]
    
    #print(f"input_string-->{input_string}")
    #print(f"message-->{message}")
    #print(f"phi_format-->{phi_format}")
    #print(f"type(message)-->{type(message)}")
    
    return phi_format

# extract txt files in folder_path to dict with all the subprompts for task, ctr, statement and answer description
def extract_lines_to_dict(folder_path, task, 
                          task_w_self_reasoning=False,
                          task_w_highlight=False,
                          task_w_one_shot=False):

    #task = folder_path.split('_')[0]
    if task == 'SemEval':
        print(f"hre")
        ordered_filenames = ['task_description', 'ctr_description', 'statement_description', 'answer_description']

        if task_w_self_reasoning == True:
            ordered_filenames = ['task_description', 'ctr_description', 'statement_description', 'self_A', 'self_B', 'self_C', 'answer_description']

        if task_w_highlight == True:
            ordered_filenames = ['task_description', 'ctr_description', 'statement_description', 'highlight_description', 'answer_description']
        if task_w_self_reasoning==True and task_w_highlight==True:
            print(f"NOT VALID SELECTION of task_w_self_reasoning and task_w_highlight. EXITING")
            sys.exit()

    elif task == 'ContractNLI':
        ordered_filenames = ['task_description', 'doc_description', 'statement_description', 'answer_description']
        if task_w_highlight == True:
            ordered_filenames = ['task_description', 'doc_description', 'statement_description', 'highlight_description', 'answer_description']

    elif task == 'CSQA':
        ordered_filenames = ['task_description', 'answer_description']
    elif task == 'MEDIQASUM':
        if task_w_one_shot == True:
            ordered_filenames = ['task_description', 'example_description', 'dialog_description', 'answer_description']
        else:
            print(f"MEDIQASUM only with one_shot flag true")
            sys.exit()
    elif task == 'Evo_prompts':
        ordered_filenames = ['mutation_prompts', 'combination_prompts']
    elif task == 'hyper_mutation':
        ordered_filenames = ['mutation_prompts']
    elif task == 'hyper_crossover':
        ordered_filenames = ['combination_prompts']
    else:
        print(f"At extract_lines_to_dict: '{task}' is not a valid task name!")
        sys.exit()


    # self_reasoning case
    if task_w_self_reasoning == True:
        ordered_filenames += ['self_A', 'self_B', 'self_C']
    
    # add .txt
    ordered_filenames = [i+'.txt' for i in ordered_filenames]
    files_dict = {}
    # Iterate through each file in the predefined order
    for file_name in ordered_filenames:
        if file_name.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Read lines from the file if it exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    content = file.read()

                # Check if the file should be split using markers
                if '0->' in content:
                    # Regex to capture blocks between 'N->' and '----------'
                    pattern = re.compile(r'\d+->(.*?)----------', re.DOTALL)
                    segments = pattern.findall(content)
                    # Strip each segment of whitespace and store it
                    files_dict[file_name[:-4]] = [segment.strip() for segment in segments]
                else:
                    lines = content.splitlines()
                    files_dict[file_name[:-4]] = [line.strip() for line in lines]
            else:
                print(f"Warning: {file_path} does not exist.")
    return files_dict


# function to extract SemEval data to a list of dictionaries with the 
# id's, 'statement', 'primary_evidence', 'label' and  'secondary_evidence' if it existss
# based on code from https://aclanthology.org/2023.semeval-1.137.pdf
def extract_SemEval_data(folder = 'DATASETS/SemEval_data', 
                         type = 'dev', 
                         extract_examples = False,
                         use_retrieves_sentences_files = True,
                         retrieve_sentences = True,
                         save_retrieved_sentences = True):

    file_path = os.path.join(folder, f"{type}_w_retrieved.json")
    if use_retrieves_sentences_files == True and os.path.exists(file_path):
        # Load from a JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)
        print(f"Used data with already retrieved examples from {file_path}")
        return data_list

    type_no_extension = type
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
    
    # most similar sentences to the statement from each CTR's section
    if retrieve_sentences == True:
        for i in tqdm(range(len(data_expanded)), desc='Retrieving...'):
            base = ' '
            primary_sentences = []
            for s in data_expanded[i]["primary_evidence"]:
                if s.endswith(': ') or s.endswith(':'):
                    base = s
                    continue
                primary_sentences.append(base + s)

            embeddings = embed_texts([data_expanded[i]['statement']]+ primary_sentences)
            similarities = cos_sim(embeddings[:1], embeddings[1:])[0]

            # select 2 most similar bits
            top_indices = np.argsort(similarities)[-2:].tolist()[::-1]
            #print(f"top_indices-->{top_indices}")
            top_sentences = [primary_sentences[idx] for idx in top_indices]
            data_expanded[i]['retrieved_primary_sentence'] = top_sentences
            
            
            if "secondary_evidence" in data_expanded[i].keys():
                base = ' '
                secondary_sentences = []
                for s in data_expanded[i]["secondary_evidence"]:
                    if s.endswith(': ') or s.endswith(':'):
                        base = s
                        continue
                    secondary_sentences.append(base + s)

                embeddings = embed_texts([data_expanded[i]['statement']]+ secondary_sentences)
                similarities = cos_sim(embeddings[:1], embeddings[1:])[0]

                # select 2 most similar bits
                top_indices = np.argsort(similarities)[-2:].tolist()[::-1]
                #print(f"top_indices-->{top_indices}")
                top_sentences = [secondary_sentences[idx] for idx in top_indices]
                data_expanded[i]['retrieved_secondary_sentence'] = top_sentences

        if save_retrieved_sentences == True:
            save_path = os.path.join(folder, f"{type_no_extension}_w_retrieved.json")
            with open(save_path, 'w') as file:
                json.dump(data_expanded, file)
            print(f"Examples with retreival svaed to {save_path}")


    if extract_examples == True:
        SUPER_current = []
        SUPER_correponding = []
        for i in range(len(data_expanded)):
            print(f"\n\ni-->{i}\n\n")
            for_similarities = []
            id_list = []
            ctr_list = []
            sec_ctr_list = []
            stat_list = []
            ans_list = []
            for example in data_expanded:
                # criterion for consideration for nearest example
                if ('secondary_evidence' in data_expanded[i].keys()) == ('secondary_evidence' in example.keys()) and example['id'] != data_expanded[i]['id']:
                    for_similarities.append("\n".join(example['primary_evidence']))
                    id_list.append(example['id'])
                    ctr_list.append(example['primary_evidence'])
                    stat_list.append(example['statement'])
                    ans_list.append(example['label'])
                    if 'secondary_evidence' in example.keys():
                        sec_ctr_list.append(example['secondary_evidence'])

            # embeddings based on primary evidence only
            aaa = ['\n'.join(data_expanded[i]['primary_evidence'])] + for_similarities
            print(f"aaa->{aaa}")
            embeddings = embed_texts(['\n'.join(data_expanded[i]['primary_evidence'])] + for_similarities)
            similarities = cos_sim(embeddings[:1], embeddings[1:])
            print(f"similarities-->{similarities}")
            closest_index = np.argmax(similarities)
            print(f"closest_index-->{closest_index}")
            similarity_score = similarities[0][closest_index]
            print(f"similarity_score-->{similarity_score}")

            print(f"len(similarities[0])-->{len(similarities[0])}")
            print(f"len(ctr_list)-->{len(ctr_list)}")
            print(f"len(sec_ctr_list)-->{len(sec_ctr_list)}")  
            print(f"len(stat_list)-->{len(stat_list)}")
            print(f"len(ans_list)-->{len(ans_list)}")

            print(f"ctr_list[closest_index]-->{ctr_list[closest_index]}")
            print(f"data_expanded[i]['primary_evidence']-->{data_expanded[i]['primary_evidence']}")

            # add most similar example
            data_expanded[i]['example_primary_evidence'] = ctr_list[closest_index]
            if 'secondary_evidence' in data_expanded[i].keys():
                data_expanded[i]['example_secondary_evidence'] = sec_ctr_list[closest_index]
            data_expanded[i]['example_statement'] = stat_list[closest_index]
            data_expanded[i]['example_answer'] = ans_list[closest_index]
            SUPER_current.append(data_expanded[i]['id'])
            SUPER_correponding.append(id_list[closest_index])

        # Check if the lists are of the same length
        assert len(SUPER_current) == len(SUPER_correponding), "UUIDs and predictions lists must have the same length."

        # Combine the lists into a dictionary
        data = {og: {"1-shot example": one_shot} for og, one_shot in zip(SUPER_current, SUPER_correponding)}

        # Save the data to a JSON file
        file_name = 'DATASETS/SemEval_data/' + "semeval_one_shot.json"
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    
    return data_expanded

# based on code from https://github.com/jonathanherzig/commonsenseqa/blob/master/esim/reader_csqa.py
# function to extract 
def extract_CSQA_data(file_path = 'DATASETS/CSQA_data', type='dev'):
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
def extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                             type = 'dev',
                             use_retrieves_sentences_files = True,
                             retrieve_sentences = False,
                             save_retrieved_sentences = False,
                             ):

    file_path = os.path.join(folder, f"{type}_w_retrieved.json")
    if use_retrieves_sentences_files == True and os.path.exists(file_path):
        # Load from a JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)
        print(f"Used data with already retrieved examples from {file_path}")
        return data_list

    type_no_extension = type
    type += '.json'
    split = type
    data = json.load(open(f"{folder}/{split}"))
    #print(f"stringaroni-->{folder}/{split}")
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
        spans = doc['spans']
        #print(f"text-->{text}")

        #print(f"doc[0]['annotation_sets'][0]['annotations']-->{doc['annotation_sets'][0]['annotations']}")
        for stat_name in doc['annotation_sets'][0]['annotations']:

            stat = statements[stat_name]
            #print(f"stat-->{stat}")

            label = doc['annotation_sets'][0]['annotations'][stat_name]['choice']

            # if only dealing with spans
            if label != 'NotMentioned':
                spans_index = doc['annotation_sets'][0]['annotations'][stat_name]['spans']
                #print(f"label-->{label}")

                # add to data_expanded, (each text has several statements associated with it)
                temp = {}
                temp['text'] = text
                temp["statement"] = stat
                temp["label"] = label
                temp["spans"] = spans
                temp["spans_index"] = spans_index
                data_expanded.append(temp)
    
    if retrieve_sentences == True:
        prev_contract = 0
        sentences = []
        for i in tqdm(range(len(data_expanded)), desc='Retrieving...'):
            
            
            if prev_contract != data_expanded[i]['text']:
                for s in data_expanded[i]["spans"]:
                    sentences.append(data_expanded[i]["text"][s[0]:s[1]])
                    #print(f"sentences-->{sentences}")
                
                sentences_embeddings = embed_texts(sentences)

            statement_embedding = embed_texts([data_expanded[i]['statement']])

            embeddings = np.vstack((statement_embedding, sentences_embeddings))
            
            similarities = cos_sim(embeddings[:1], embeddings[1:])[0]
            #print(f"similarities-->{similarities}")
            # Get indices of the 3 largest similarities
            top_indices = np.argsort(similarities)[-3:].tolist()[::-1]
            print(f"top_indices-->{top_indices}")
            # Retrieve the sentences corresponding to the top 3 indices
            #print(f"len(sentences)-->{len(sentences)}")
            top_sentences = [sentences[idx] for idx in top_indices]
            print(f"statement-->{data_expanded[i]['statement']}")
            print(f"top_sentences-->{top_sentences}")
            data_expanded[i]['retrieved_sentences'] = top_sentences
            prev_contract = data_expanded[i]['text']
        
        if save_retrieved_sentences == True:
            save_path = os.path.join(folder, f"{type_no_extension}_w_retrieved.json")
            with open(save_path, 'w') as file:
                json.dump(data_expanded, file)
            print(f"Examples with retreival svaed to {save_path}")

    return data_expanded

# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('Entailment' or 'Contradiction')
# based on code from https://aclanthology.org/2023.semeval-1.137.pdf
def prompt_creation_semeval(data_expanded, 
                            task_description, 
                            ctr_description, 
                            statement_description, 
                            answer_description,
                            model,
                            task_w_self_reasoning=False,
                            task_w_highlight=False,
                            task_w_one_shot=False, 
                            example_description=None,
                            highlight_description = '',
                            self_A = '',
                            self_B = '',
                            self_C = '',
                            ):

    if task_w_self_reasoning==True and self_A != '' and self_A != ' ':
        labels, predictions = prompt_preds_semeval_self(data_expanded=data_expanded, 
                                                        task_description=task_description, 
                                                        ctr_description=ctr_description, 
                                                        statement_description=statement_description, 
                                                        self_A=self_A, 
                                                        self_B=self_B, 
                                                        self_C=self_C, 
                                                        #model=model, 
                                                        #tokenizer=tokenizer, 
                                                        #trie=trie
                                                        )
        
        return labels, predictions

    labels = []
    samples = []

    for sample in tqdm(data_expanded, desc='Creating Prompts'):
        prompt = task_description + '\n\n'

        # one shot case
        if task_w_one_shot == True and example_description != None:

            example_primary_evidence = "\n".join(sample['example_primary_evidence'])
            example_sentence = f"""{prompt}Primary Trial\n"{example_primary_evidence}" """
            example_secondary_evidence = sample.get("example_secondary_evidence")
            if example_secondary_evidence:
                example_secondary_evidence = "\n".join(sample['example_secondary_evidence'])
                example_sentence = f"""{example_sentence}\n\nSecondary Trial\n"{example_secondary_evidence}" """

            example_stat = "".join(sample['example_statement'])
            example_sentence = f"""Example CTR: "{example_sentence}"\n\nExample Statement: "{example_stat}"\n\nExample ANSWER: {sample['example_answer']}"""


        prompt = prompt + ctr_description + '\n\n'
        primary_evidence = "\n".join(sample['primary_evidence'])
        sentence = f"""{prompt}Primary Trial\n"{primary_evidence}" """
        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary_evidence = "\n".join(sample['secondary_evidence'])
            sentence = f"""{sentence}\n\nSecondary Trial\n"{secondary_evidence}" """
        #input_text = get_input_text(sentence, sample['statement'])
        stat = "".join(sample['statement'])

        sentence = f"""{sentence}\n\n{statement_description}\n\n"{stat}"\n\n"""

        # if using retrieved highlight, just adds them before the answer
        if task_w_highlight == True and highlight_description != '' and highlight_description != ' ':
            retrieved_primary = "\n".join(sample['retrieved_primary_sentence'])
            sentence = sentence = f"""{sentence}{highlight_description}\n\nPimary CTR: "{retrieved_primary}"\n\n"""
            if secondary_evidence:
                retrieved_secondary = "\n".join(sample['retrieved_secondary_sentence'])
                sentence = sentence = f"""{sentence}Secondary CTR: "{retrieved_secondary}"\n\n"""

        sentence = f"""[INST]{sentence}{answer_description}[/INST]ANSWER:"""

        # conversion necessary for phi3 model
        if 'Phi3' in str(model):
            sentence = convert_text_mistral_phi3(sentence)

        labels.append(sample["label"])
        samples.append(sentence)
        #print(f"SENTENCE-->{sentence}")

    return samples, labels


def batch_inference(all_prompts, model, tokenizer, trie, batch_size=16):

    # Tokenize all prompts
    encodings = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')

    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds = []

    # Process in batches
    for batch in tqdm(dataloader, desc='Evaluating prompts in batches'):
        input_ids, attention_mask = batch
        prompt_length = input_ids.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, 
                                     attention_mask=attention_mask, 
                                     pad_token_id=tokenizer.eos_token_id, 
                                     max_new_tokens=3,
                                     prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))

        for output in outputs:
            new_tokens = output[prompt_length:]
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            #print(f"PRED-->{pred}")
            preds.append(pred)
            #print(f"INFERENCE-->{tokenizer.decode(output, skip_special_tokens=False)}")

    predictions, _ = convert_preds_from_yesno(preds)

    return predictions


def prompt_preds_semeval(data_expanded, 
                         task_description, 
                         ctr_description, 
                         statement_description, 
                         answer_description,
                         model, tokenizer, trie,
                         task_w_self_reasoning=False,
                         task_w_highlight=False,
                         task_w_one_shot=False, 
                         example_description=None,
                         highlight_description = '',
                         self_A = '',
                         self_B = '',
                         self_C = '',
                         ):
    '''
    samples, labels = prompt_creation_semeval(data_expanded=data_expanded, 
                                            task_description=task_description, 
                                            ctr_description=ctr_description, 
                                            statement_description=statement_description, 
                                            answer_description=answer_description,
                                            model=model,
                                            task_w_self_reasoning=task_w_self_reasoning,
                                            task_w_highlight=task_w_highlight,
                                            task_w_one_shot=task_w_one_shot, 
                                            example_description=example_description,
                                            highlight_description = highlight_description,
                                            self_A = self_A,
                                            self_B = self_B,
                                            self_C = self_C,
                                            )
    
    predictions = batch_inference(all_prompts = samples, model=model, tokenizer=tokenizer, trie=trie)
    
    return labels, predictions
    '''
    print(f"self_A-->{self_A}")
    if task_w_self_reasoning==True and self_A != '' and self_A != ' ':
        print(f"COM SELF REASONING")

        labels, predictions = prompt_preds_semeval_self(data_expanded=data_expanded, 
                                                        task_description=task_description, 
                                                        ctr_description=ctr_description, 
                                                        statement_description=statement_description, 
                                                        self_A=self_A, 
                                                        self_B=self_B, 
                                                        self_C=self_C, 
                                                        model=model, 
                                                        tokenizer=tokenizer, 
                                                        trie=trie)
        
        return labels, predictions

    #print(f"SEM SELF REASONING")
    labels = []
    preds = []
    flag = 0

    for sample in tqdm(data_expanded, desc='Evaluating prompts for individual'):
        prompt = task_description + '\n\n'

        # one shot case
        if task_w_one_shot == True and example_description != None:

            example_primary_evidence = "\n".join(sample['example_primary_evidence'])
            example_sentence = f"""{prompt}Primary Trial\n"{example_primary_evidence}" """
            example_secondary_evidence = sample.get("example_secondary_evidence")
            if example_secondary_evidence:
                example_secondary_evidence = "\n".join(sample['example_secondary_evidence'])
                example_sentence = f"""{example_sentence}\n\nSecondary Trial\n"{example_secondary_evidence}" """

            example_stat = "".join(sample['example_statement'])
            example_sentence = f"""Example CTR: "{example_sentence}"\n\nExample Statement: "{example_stat}"\n\nExample ANSWER: {sample['example_answer']}"""


        prompt = prompt + ctr_description + '\n\n'
        primary_evidence = "\n".join(sample['primary_evidence'])
        sentence = f"""{prompt}Primary Trial\n"{primary_evidence}" """
        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary_evidence = "\n".join(sample['secondary_evidence'])
            sentence = f"""{sentence}\n\nSecondary Trial\n"{secondary_evidence}" """
        #input_text = get_input_text(sentence, sample['statement'])
        stat = "".join(sample['statement'])

        sentence = f"""{sentence}\n\n{statement_description}\n\n"{stat}"\n\n"""

        # if using retrieved highlight, just adds them before the answer
        if task_w_highlight == True and highlight_description != '' and highlight_description != ' ':
            retrieved_primary = "\n".join(sample['retrieved_primary_sentence'])
            sentence = sentence = f"""{sentence}{highlight_description}\n\nPimary CTR: "{retrieved_primary}"\n\n"""
            if secondary_evidence:
                retrieved_secondary = "\n".join(sample['retrieved_secondary_sentence'])
                sentence = sentence = f"""{sentence}Secondary CTR: "{retrieved_secondary}"\n\n"""

        sentence = f"""[INST]{sentence}{answer_description}[/INST]\n\nANSWER:"""

        labels.append(sample["label"])

    

        # conversion necessary for phi3 model
        if 'Phi3' in str(model):
            sentence = convert_text_mistral_phi3(sentence)

        prompt = tokenizer.encode(sentence, return_tensors="pt", return_attention_mask=True).to('cuda')

        prompt_length = prompt[0].shape[0]
        with torch.inference_mode():
            output = model.generate(prompt, 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens = 3,
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))


        #print(f"SEMEVAL inference-->{tokenizer.decode(output[0], skip_special_tokens=False)}")
        #print(f"TRUE LABEL-->{sample['label']}")

        new_tokens = output[0, prompt_length:]

        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        #print(f"pred-->{pred}")
        preds.append(pred)

    predictions, _ = convert_preds_from_yesno(preds)
    return labels, predictions
    

    

def prompt_creation_semeval_self(data_expanded, task_description, ctr_description, statement_description,
                                 self_A, self_B, self_C, model, tokenizer):
    samples = []
    for sample in tqdm(data_expanded, desc='creating self reasoning prompts'):
        primary ="\n".join(sample['primary_evidence'])
        text = f"""{task_description}\n\n{ctr_description}\n\nPrimary Trial: "{primary}"\n\n """

        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary ="\n".join(sample['primary_evidence'])
            text = f"""{text} Secondary Trial: "{secondary}"\n\n """

        text_self = f"""[INST]{text}{statement_description}\n\n"{sample['statement']}"\n\n{self_A}[/INST]\n\nANSWER: """

        # Tokenize input and generate attention mask
        prompt_self = tokenizer.encode(text_self, return_tensors="pt").to('cuda')
        prompt_length = prompt_self[0].shape[0]
        
        try:
            # to improve efficiency
            with torch.inference_mode():
                output = model.generate(prompt_self, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100, do_sample=True, num_beams = 3)
        except:
            output = ''
            prompt_length = 0

        #print(f"prompt_length-->{prompt_length}")
        new_tokens = output[0, prompt_length:]
        reflection = tokenizer.decode(new_tokens, skip_special_tokens=True)
        #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->\n{tokenizer.decode(output[0], skip_special_tokens=False)}")

        text_w_reflection = f"""[INST]{text}{statement_description}\n\n"{sample['statement']}"\n\n{self_B}\n\n"{reflection}"\n\n{self_C}[/INST]\n\nANSWER: """

        temp = {"text":text_w_reflection, "label":sample['label']}
        samples.append(temp)

    return samples


def prompt_preds_semeval_self(data_expanded, task_description, ctr_description, statement_description,
                                 self_A, self_B, self_C, model, tokenizer, trie):

    labels = []
    preds = []
    flag=0
    for sample in tqdm(data_expanded, desc='Self Reasoning'):
        primary = "\n".join(sample['primary_evidence'])
        text = f"""{task_description}\n\n{ctr_description}\n\nPrimary Trial: "{primary}"\n\n """

        secondary_evidence = sample.get("secondary_evidence")
        if secondary_evidence:
            secondary = "\n".join(sample['secondary_evidence'])
            text = f"""{text} Secondary Trial: "{secondary}"\n\n """

        common_text = f"""[INST]{text}{statement_description}\n\n"{sample['statement']}"\n\n"""
        text_self = f"""{common_text}{self_A}[/INST]\n\nANSWER: """

        #if 'Phi3' in str(model):
            #common_text = convert_text_mistral_phi3(common_text)

        #cached = tokenizer.encode(common_text, return_tensors="pt")
        #cached_outputs = model(cached, return_dict=True,)
        #cached_outputs.past_key_values = [[y[:, :, :-1] for y in x] for x in cached_outputs.past_key_values]

        # conversion necessary for phi3 model
        if 'Phi3' in str(model):
            text_self = convert_text_mistral_phi3(text_self)

        prompt = tokenizer.encode(text_self, return_tensors="pt", return_attention_mask=True).to('cuda')

        prompt_length = prompt[0].shape[0]

        with torch.inference_mode():
            output = model.generate(prompt, 
                                    #past_key_values=cached_outputs.past_key_values, 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=300, use_cache=True, do_sample=True, num_beams = 3)

        #print(f"prompt_length-->{prompt_length}")
        new_tokens = output[0, prompt_length:]
        reflection = tokenizer.decode(new_tokens, skip_special_tokens=True)
        #print(f"tokenizer.decode(output[0], skip_special_tokens=False)-->\n{tokenizer.decode(output[0], skip_special_tokens=False)}")

        text_w_reflection = f"""{common_text}{self_B}\n\n"{reflection}"\n\n{self_C}[/INST]\n\nANSWER: """
        #print(f"text_w_reflection-->{text_w_reflection}\n\n\n")

        # to improve memory usage of gpu
        del prompt, output
        # Clear GPU cache
        torch.cuda.empty_cache()

        # conversion necessary for phi3 model
        if 'Phi3' in str(model):
            text_w_reflection = convert_text_mistral_phi3(text_w_reflection)

        prompt = tokenizer.encode(text_w_reflection, return_tensors="pt", return_attention_mask=True).to('cuda')

        #prompt = tokenizer.encode(text_w_reflection, return_tensors="pt").to('cuda')
        prompt_length = prompt[0].shape[0]
        #print(f"prompt_length-->{prompt_length}")

        with torch.inference_mode():
            output = model.generate(prompt, 
                                    #past_key_values=cached_outputs.past_key_values, 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=6, use_cache=True,
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
        #if flag == 0:
            #print(f"SEMEVAL inference-->{tokenizer.decode(output[0], skip_special_tokens=False)}")
            #print(f"TRUE LABEL-->{sample['label']}")
            #flag=1

        new_tokens = output[0, prompt_length:]

        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds.append(pred)
        labels.append(sample["label"])

        # to improve memory usage of gpu
        del prompt, output
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        predictions, _ = convert_preds_from_yesno(preds)

    return labels, predictions

# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('A', 'B', 'C', 'D' or 'E')
def prompt_creation_csqa(data_expanded, task_description, answer_description):
    samples = []
    letters = ['A', 'B', 'C', 'D', 'E']
    for sample in data_expanded:
        prompt = task_description + '\n' 
        sentence = f"""{prompt}\n"{sample['question']}"""

        option_list = ''
        for i, j in zip(sample['choice'], letters):
            option = f"{j} - {i}\n"
            option_list += option

        sentence = f"""[INST]{sentence}\n{option_list}"\n{answer_description}[/INST]\n\nANSWER:"""
        temp = {"text":sentence, "label":sample['label']}
        samples.append(temp)

    return samples

# function to create list of dictionaries with:
# text: text to prompt the LLM, made from the subprompts and the data
# label: true label ('Entailment' or 'Contradiction' or 'NotMentioned')
def prompt_creation_contractnli(data_expanded, task_description, doc_description, statement_description, answer_description):
    samples = []
    for sample in data_expanded:
        prompt_wo_statement = f"[INST]{task_description}\n\n{doc_description}\n\n{sample['text']}\n\n{statement_description}\n\n"
        prompt = f"{prompt_wo_statement}{sample['statement']}\n\n{answer_description}[/INST]\n\nANSWER: "
        temp = {"text":prompt, "label":sample['label'], "text2cache" : prompt_wo_statement}
        samples.append(temp)

    return samples

def prompt_creation_contractnli_span(data_expanded, task_description, doc_description, statement_description, answer_description):
    samples = []
    for sample in data_expanded:
        if sample['label'] != 'NotMentioned':
            before_nda = f"""[INST]{task_description}\n\n{doc_description}\n """
            # add spans
            marker = 0
            for i in sample['spans_index']:
                before_nda += f"\n"
                if marker == 0:
                    before_nda += f""" " """
                    marker = 1
                
                before_nda += f"{sample['text'][sample['spans'][i][0]:sample['spans'][i][1]]}"
            
            before_nda += f""" " \n\n"""
            prompt_wo_statement = f"{before_nda}{statement_description}\n\n"
            prompt = f"""{prompt_wo_statement}"{sample['statement']}"\n\n{answer_description}[/INST]\n\nANSWER: """
            temp = {"text":prompt, "label":sample['label'], "text2cache" : prompt_wo_statement}
            samples.append(temp)

    return samples


def prompt_preds_contractnli_span(data_expanded, task_description, doc_description, statement_description, answer_description,
                                  model, tokenizer, trie):
    labels = []
    preds = []
    per_doc_labels = []
    per_doc_preds = []

    doc = data_expanded[0]['text']
    single_doc_preds = []
    single_doc_labels = []

    for sample in tqdm(data_expanded, desc='Evaluating prompts for individual'):
        if sample['label'] != 'NotMentioned':
            before_nda = f"""[INST]{task_description}\n\n{doc_description}\n """
            #print(f"before_nda-->{before_nda}")
            # add spans
            marker = 0
            for i in sample['spans_index']:
                before_nda += f"\n"
                if marker == 0:
                    before_nda += f""" " """
                    marker = 1
                
                before_nda += f"{sample['text'][sample['spans'][i][0]:sample['spans'][i][1]]}"
            
            #print(f"before_nda-->{before_nda}")
            before_nda += f""" " \n\n"""
            prompt_wo_statement_text = f"{before_nda}{statement_description}\n\n"
            prompt_text = f"""{prompt_wo_statement_text}"{sample['statement']}"\n\n{answer_description}[/INST]\n\nANSWER: """
            #print(f"prompt-->{prompt}")

            cached_text = 0
            #torch.cuda.empty_cache()
            labels.append(sample["label"])

            # cache part of input that does not change from prompt to prompt
            if prompt_wo_statement_text == cached_text:
                #print(f"equal text")
                pass
            else:
                #print(f"new text")
                cached_text = prompt_wo_statement_text
                cached = tokenizer.encode(cached_text, return_tensors="pt")
                with torch.inference_mode():
                    cached_outputs = model(cached, return_dict=True,)
                cached_outputs.past_key_values = [[y[:, :, :-1] for y in x] for x in cached_outputs.past_key_values]

            # conversion necessary for phi3 model
            if 'Phi3' in str(model):
                prompt_text = convert_text_mistral_phi3(prompt_text)

            prompt = tokenizer.encode(prompt_text, return_tensors="pt", return_attention_mask=True).to('cuda')

            prompt_length = prompt[0].shape[0]
            #print(f"prompt_length-->{prompt_length}")
            with torch.inference_mode():
                output = model.generate(prompt, past_key_values=cached_outputs.past_key_values, 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=6, use_cache=True,
                                    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))

            # Decode only the newly generated tokens
            # Skip the input tokens by starting the slice at input_length
            new_tokens = output[0, prompt_length:]

            #print(f"tokenizer.decode(new_tokens)-->{tokenizer.decode(new_tokens)}")
            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)

            preds.append(pred)

            # to calculate f1 scores according to the paper (macro avg micro avgs per document)
            if sample['text'] == doc:
                single_doc_preds.append(pred)
                single_doc_labels.append(sample["label"])

            else:
                doc = sample['text']
                single_doc_preds, _ = convert_preds_from_yesno(single_doc_preds)
                per_doc_preds.append(single_doc_preds)
                per_doc_labels.append(single_doc_labels)
                single_doc_preds = []
                single_doc_labels = []
                single_doc_preds.append(pred)
                single_doc_labels.append(sample["label"])
    
    single_doc_preds, _ = convert_preds_from_yesno(single_doc_preds)
    per_doc_preds.append(single_doc_preds)
    per_doc_labels.append(single_doc_labels)

    predictions, _ = convert_preds_from_yesno(preds)
    
    return labels, predictions, per_doc_labels, per_doc_preds #listas de dimensão (n_docs) 

def extract_MEDIQASUM_data(folder_name='DATASETS/MEDIQASUM_data', 
                           type = 'valid', 
                           used_retrieved_file = True,
                           retrieve_similar_examples = True,
                           save_retrieved = False
                           ):
    file_path = os.path.join(folder_name, f"{type}_w_retrieved.json")
    if used_retrieved_file == True and os.path.exists(file_path):
        # Load from a JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)
        print(f"Used data with already retrieved examples from {file_path}")
        return data_list

    
    # Construct the full file path
    file_path = os.path.join(folder_name, f"{type}.csv")
    print(f"Retrieving examples for data from {file_path}")
    # Create an empty list to hold the dictionaries
    data_list = []
    # Open the CSV file and read data into a list of dictionaries
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data_list.append(row)

    # example retrieval part
    # retrieving always from the train set
    if retrieve_similar_examples == True:

        # train data path
        train_file_path = os.path.join(folder_name, f"train.csv")
        # Create an empty list to hold the dictionaries
        train_data_list = []
        # Open the CSV file and read data into a list of dictionaries
        with open(train_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                train_data_list.append(row)

        # extract train_data
        for i in tqdm(range(len(data_list)), desc= 'Retrieving examples'):
            #notes_list = []
            train_dialogue_list = []
            for example in train_data_list:
                # embed dialogues to then select most similar examples based on this
                if data_list[i]['dataset'] == example['dataset'] and data_list[i]['encounter_id'] != example['encounter_id']:
                    train_dialogue_list.append(example['dialogue'])
                    #print(f"example['dialogue']-->{example['dialogue']}\n\n\n")

            embeddings = embed_texts([data_list[i]['dialogue']]+ train_dialogue_list)
            similarities = cos_sim(embeddings[:1], embeddings[1:])
            #print(f"similarities-->{similarities}")
            closest_index = np.argmax(similarities)
            #print(f"closest_index-->{closest_index}")
            similarity_score = similarities[0][closest_index]
            #print(f"RETRIEVAL->similarity_score-->{similarity_score}")
            # add most similar example
            data_list[i]['retrieved_example_note'] = train_data_list[closest_index]['note']
            #print(f"data_list[i]['retrieved_example_note']-->{data_list[i]['retrieved_example_note']}")


        if save_retrieved == True:
            # Save to a JSON file
            save_path = os.path.join(folder_name, f"valid_w_retrieved.json")
            with open(save_path, 'w') as file:
                json.dump(data_list, file)
            print(f"Examples with retreival svaed to {save_path}")
    
    return data_list

# randomly select example
def random_example_retrieval(sample, data_expanded):
    for example in data_expanded:
        if sample['dataset'] == example['dataset'] and sample['encounter_id'] != example['encounter_id']:
            note = example['note']
            return note
    return note

def embed_texts(texts, model_name='Alibaba-NLP/gte-large-en-v1.5'):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(texts)
    return embeddings

# retrieval using embedding model
def similar_example_retrieval(sample, data_expanded, filter_by_dataset=True):

    notes_list = []
    for example in data_expanded:
        if sample['dataset'] == example['dataset'] and sample['encounter_id'] != example['encounter_id']:
            notes_list.append(example['note'])

    embeddings = embed_texts([sample['note']]+ notes_list)
    similarities = cos_sim(embeddings[:1], embeddings[1:])
    closest_index = np.argmax(similarities)
    similarity_score = similarities[closest_index]
    print(f"RETRIEVAL - similar_example_retrieval function - ->similarity_score-->{similarity_score}")
    return notes_list[closest_index]

def prompt_preds_mediqasum(data_expanded, task_description, example_description, dialog_description, answer_description,
                              model, tokenizer):

    labels = []
    preds = []

    for sample in tqdm(data_expanded):

        prompt = task_description + '\n\n' + example_description + '\n\n'
        ## VERIFICAR ISTO
        if 'retrieved_example_note' in sample.keys():
            example = sample['retrieved_example_note']
        else:
            example = random_example_retrieval(sample, data_expanded)
        sentence = f"""{prompt}Example Note:\n"{example}" """
        dialogue = "".join(sample['dialogue'])
        sentence = f"""[INST]{sentence}\n\n{dialog_description}\n\n"{dialogue}"\n\n{answer_description}[/INST]\n\nClinical Note:"""
        labels.append(sample["note"])

        # conversion necessary for phi3 model
        if 'Phi3' in str(model):
            sentence = convert_text_mistral_phi3(sentence)
            print(f"messages prompts-->{sentence}\n\n\n\n\n\n\n\n")

        prompt = tokenizer.encode(sentence, return_tensors="pt", return_attention_mask=True).to('cuda')
            
        prompt_length = prompt[0].shape[0]
        with torch.inference_mode():
            
            output = model.generate(prompt, 
                                    pad_token_id=tokenizer.eos_token_id, 
                                    max_new_tokens=1200,
                                    do_sample=True,
                                    num_beams = 3
                                    )  

        # Decode only the newly generated tokens
        # Skip the input tokens by starting the slice at input_length
        new_tokens = output[0, prompt_length:]

        pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
        preds.append(pred)

        del prompt, output
        torch.cuda.empty_cache()

    return labels, preds


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
def load_model(checkpoint = "microsoft/Phi-3-mini-128k-instruct",
               quantized = True):
    
    #print(f"load MODEL 1-")
    torch.cuda.empty_cache()
    #print(f"load MODEL 2-")
    if 'Phi-3' in checkpoint:
        #print(f"load MODEL 4-")
        
        config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                )
        # try to load with flahs attention if gpu allows it
        #print(f"load MODEL 5-")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint, 
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
        #print(f"load MODEL 6-")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
        #print(f"load MODEL 7-")
        return model, tokenizer

    else:
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
def mutate_prompt(prompt, mutation_prompt, model, tokenizer):

    # case with empty string in self reflection prompts
    if prompt == '' or prompt == ' ':
        return ''
    instruction = '[INST]' + mutation_prompt + """\n\nINSTRUCTION:" """ + prompt + """ "[/INST]""" + "\n\nNEW INSTRUCTION: "
    #print(f"instruction-->{instruction}")

    # conversion necessary for phi3 model
    if 'Phi3' in str(model):
        instruction = convert_text_mistral_phi3(instruction)

    prompt = tokenizer.encode(instruction, return_tensors="pt", return_attention_mask=True).to('cuda')

    prompt_length = prompt[0].shape[0]

    try:
        # to improve efficiency
        with torch.inference_mode():
            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_new_tokens=400, do_sample=True, num_beams = 3)
    except:
        output = ''

    new_tokens = output[0, prompt_length:]
    mutated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #print(f"MUTATE PROMPT)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")

    if mutated.startswith('"') and mutated.endswith('"'):
        # Remove the quotes
        return mutated[1:-1]

    return mutated


# function to combine prompts using an LLM
# takes a combination prompt (asking to join two instructions) and two subprompt (to be combined), outputs the NEW combined subprompt
def crossover_prompts(prompt_1, prompt_2, combination_prompt, model, tokenizer):

    # if oen of the prompts is an empty string, randomly choose one of the two to be returned
    if prompt_1 == '' or prompt_1 == ' ' or prompt_2 == '' or prompt_2 == ' ':
        if random.random() < 0.5:
            return prompt_1
        else:
            return prompt_2
    
    instruction = '[INST]' + combination_prompt + "\n\nINSTRUCTION 1: " + """ " """ + prompt_1 + """ " """ + "\n\nINSTRUCTION 2: " + """ " """ + prompt_2 + """ " """ + '[/INST]' + "\n\nNEW INSTRUCTION: "

    # conversion necessary for phi3 model
    if 'Phi3' in str(model):
        instruction = convert_text_mistral_phi3(instruction)

    prompt = tokenizer.encode(instruction, return_tensors="pt", return_attention_mask=True).to('cuda')

    prompt_length = prompt[0].shape[0]
    # Tokenize input and generate attention mask

    try:
        # to improve efficiency
        with torch.inference_mode():
            output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_new_tokens=400, do_sample=True, num_beams = 3)
    except:
        output = ''

    new_tokens = output[0, prompt_length:]
    combined = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #print(f"CROSSOVER PROMPT)-->{tokenizer.decode(output[0], skip_special_tokens=False)}")

    if combined.startswith('"') and combined.endswith('"'):
        # Remove the quotes
        return combined[1:-1]

    return combined

# used to limit decoding options
# given the task a set of possible answers is selected, which are then tokenized and used
# to create the MarisaTrie object
def get_Marisa_Trie(task, tokenizer):
    if task == 'SemEval' or task == 'SemEval_self':
        # IR ALTERAR A FUNÇÃO QUE depois CONVERTE PARA AS OPTIONS REAIS
        possibilities = ["YES", "NO"]
    elif task == 'CSQA':
        possibilities = ['A', 'B', 'C', 'D', 'E']
    elif task == 'ContractNLI':
        possibilities = ['UNMENTIONED', 'YES', 'NO']
        possibilities = ['YES', 'NO']
    
    encoded_possibilities = []
    for pos in possibilities:
        encoded_possibilities.append(tokenizer.encode(pos) + [tokenizer.eos_token_id])

    class MyMarisaTrie(MarisaTrie):
        def __init__(self, data): super().__init__(data)
        def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])
    trie = MyMarisaTrie(encoded_possibilities)

    return trie

# function to generate predictions for the task for a given prompt
# outputs both the predictions and the true labels
def semeval_predictions(model, tokenizer, samples, trie):

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

            pred = tokenizer.decode(new_tokens, skip_special_tokens=True)
            #print(f"pred-->{pred}")
            preds.append(pred)
    return labels, preds


# function to generate predictions for the task for a given prompt
# outputs both the predictions and the true labels
def csqa_predictions(model, tokenizer, samples, trie):

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

    labels = []
    preds = []

    cached_text = 0
    with torch.inference_mode():
        for sample in tqdm(samples, desc = f"Generating Predictions with LLM"):
            #torch.cuda.empty_cache()
            labels.append(sample["label"])

            # cache part of input that does not change from prompt to prompt
            if sample['text2cache'] == cached_text:
                #print(f"equal text")
                pass
            else:
                #print(f"new text")
                cached_text = sample['text2cache']
                cached = tokenizer.encode(cached_text, return_tensors="pt")
                cached_outputs = model(cached, return_dict=True,)
                cached_outputs.past_key_values = [[y[:, :, :-1] for y in x] for x in cached_outputs.past_key_values]


            # Tokenize input and generate attention mask
            prompt = tokenizer.encode(sample["text"], return_tensors="pt").to('cuda')
            prompt_length = prompt[0].shape[0]
            #print(f"prompt_length-->{prompt_length}")
            try:
                output = model.generate(prompt, past_key_values=cached_outputs.past_key_values, 
                                        pad_token_id=tokenizer.eos_token_id, 
                                        max_new_tokens=6, use_cache=True,
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

            preds.append(pred)

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
        if i == 'YES' or i == 'Yes' or i == 'yes' or i == 'Entailment'or i == 'ENTAILMENT':
            preds_2.append('Entailment')
        elif i == 'NO' or i == 'No' or i == 'no' or i == 'Contradiction'or i == 'CONTRADICTION':
            preds_2.append('Contradiction')
        elif i == 'Not mentioned' or i == 'Not Mentioned' or i == 'NOT MENTIONED' or i == 'UNMENTIONED':
            preds_2.append('NotMentioned')
        else:
            print('olha as labels')
            preds_2.append('Contradiction')
            no_of_not_founds += 1

    return preds_2, no_of_not_founds

# function to evaluate prompt population
# outputs a list with the scores for each prompt
# n_samples is the no. of samples where the evaluation will be done
def eval_pop(population, data_expanded, model, tokenizer, trie, n_samples, N=10, mutation_prob=0.8,# last 2 only for hyper
             only_rouge = True, # for mediqasum 
             save_preds4semeval_test = False,
             folder = None,
             task_w_one_shot = False,
             task_w_self_reasoning = False,
             task_w_highlight = False,
             ): 
    prompts = population['prompts_dict']
    task = population['task']

    if n_samples == 0 or n_samples > len(data_expanded):
        n_samples = len(data_expanded)
    
    n_pop = len(population['prompts'])
    if task == "SemEval":
        population['f1_scores'] = []
        population['confusion_matrix'] = []
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):

            if task_w_highlight == False and task_w_self_reasoning == False:
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                       task_description = prompts['task_description'][population['prompts'][i]['task_description']],
                                                       ctr_description = prompts['ctr_description'][population['prompts'][i]['ctr_description']],
                                                       statement_description = prompts['statement_description'][population['prompts'][i]['statement_description']],
                                                       answer_description = prompts['answer_description'][population['prompts'][i]['answer_description']],
                                                       model=model,
                                                       tokenizer=tokenizer,
                                                       trie=trie,
                                                       task_w_one_shot = task_w_one_shot,
                                                       task_w_highlight = task_w_highlight,
                                                       task_w_self_reasoning = task_w_self_reasoning
                                                       )
            if task_w_highlight == True and task_w_self_reasoning == False:
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                       task_description = prompts['task_description'][population['prompts'][i]['task_description']],
                                                       ctr_description = prompts['ctr_description'][population['prompts'][i]['ctr_description']],
                                                       statement_description = prompts['statement_description'][population['prompts'][i]['statement_description']],
                                                       answer_description = prompts['answer_description'][population['prompts'][i]['answer_description']],
                                                       model=model,
                                                       tokenizer=tokenizer,
                                                       trie=trie,
                                                       task_w_one_shot = task_w_one_shot,
                                                       task_w_highlight = task_w_highlight,
                                                       task_w_self_reasoning = task_w_self_reasoning,
                                                       highlight_description = prompts['highlight_description'][population['prompts'][i]['highlight_description']],
                                                       )

            if task_w_highlight == False and task_w_self_reasoning == True:
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                       task_description = prompts['task_description'][population['prompts'][i]['task_description']],
                                                       ctr_description = prompts['ctr_description'][population['prompts'][i]['ctr_description']],
                                                       statement_description = prompts['statement_description'][population['prompts'][i]['statement_description']],
                                                       answer_description = prompts['answer_description'][population['prompts'][i]['answer_description']],
                                                       model=model,
                                                       tokenizer=tokenizer,
                                                       trie=trie,
                                                       task_w_one_shot = task_w_one_shot,
                                                       task_w_highlight = task_w_highlight,
                                                       task_w_self_reasoning = task_w_self_reasoning,
                                                       self_A = prompts['self_A'][population['prompts'][i]['self_A']],
                                                       self_B = prompts['self_B'][population['prompts'][i]['self_B']],
                                                       self_C = prompts['self_C'][population['prompts'][i]['self_C']],
                                                       )

            score = f1_score(y_true=labels, y_pred=predictions, average='macro')
            print(f"score at eval-->{score}")
            population['eval'].append(score)

            # f-1 score for more detailed analysis
            f1_scores_per_class = f1_score(y_true=labels, y_pred=predictions, average=None)
            # Find unique class labels
            unique_labels = np.unique(np.concatenate((labels, predictions)))
            # Pair each unique class label with its F1-score
            label_to_f1 = dict(zip(unique_labels, f1_scores_per_class))
            population['f1_scores'].append(label_to_f1)
            # confusion matrix
            cm = confusion_matrix(y_true=labels, y_pred=predictions, labels=unique_labels)
            population['confusion_matrix'].append(cm)

            if save_preds4semeval_test == True:
                print(f"save_preds4semeval_test-->{save_preds4semeval_test}")
                uuid_list = []
                for ex in data_expanded[:n_samples]:
                    uuid_list.append(ex['id'])

                # Check if the lists are of the same length
                assert len(uuid_list) == len(predictions), "UUIDs and predictions lists must have the same length."

                # Combine the lists into a dictionary
                data = {uuid: {"Prediction": prediction} for uuid, prediction in zip(uuid_list, predictions)}

                # Save the data to a JSON file
                file_name = folder + "test_predictions.json"
                with open(file_name, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                    
                print(f"Data successfully saved to {file_name}")

    elif task == "SemEval_self":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            #print(f"i-->{i}")
            #print(f"len(prompts['task_description'])-->{len(prompts['task_description'])}")
            if prompts['self_A'][population['prompts'][i][3]] != '':

                # with caching
                labels, predictions = prompt_preds_semeval_self(data_expanded[:n_samples], 
                                                                prompts['task_description'][population['prompts'][i][0]], 
                                                                prompts['ctr_description'][population['prompts'][i][1]], 
                                                                prompts['statement_description'][population['prompts'][i][2]], 
                                                                prompts['self_A'][population['prompts'][i][3]],
                                                                prompts['self_B'][population['prompts'][i][4]],
                                                                prompts['self_C'][population['prompts'][i][5]],
                                                                model,
                                                                tokenizer, trie)
            else:
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                            prompts['task_description'][population['prompts'][i][0]], 
                                                            prompts['ctr_description'][population['prompts'][i][1]], 
                                                            prompts['statement_description'][population['prompts'][i][2]], 
                                                            'Based on the clinical trial report descriptions and the statement provided, assess the validity of the statement by carefully understanding the medical terminology and context in both the report and the statement, resolving any ambiguities or information gaps. (YES or NO response acceptable)',
                                                            model, tokenizer, trie)

            score = f1_score(y_true=labels, y_pred=predictions, average='macro')
            print(f"score-->{score}")

            population['eval'].append(score)

            if save_preds4semeval_test == True:
                print(f"save_preds4semeval_test-->{save_preds4semeval_test}")
                uuid_list = []
                for ex in data_expanded[:n_samples]:
                    uuid_list.append(ex['id'])

                # Check if the lists are of the same length
                assert len(uuid_list) == len(predictions), "UUIDs and predictions lists must have the same length."

                # Combine the lists into a dictionary
                data = {uuid: {"Prediction": prediction} for uuid, prediction in zip(uuid_list, predictions)}

                # Save the data to a JSON file
                file_name = folder + "test_predictions.json"
                with open(file_name, 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                    
                print(f"Data successfully saved to {file_name}")

    elif task == "CSQA":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            samples = prompt_creation_csqa(data_expanded, 
                                           task_description = prompts['task_description'][population['prompts'][i]['task_description']],
                                           answer_description = prompts['answer_description'][population['prompts'][i]['answer_description']],
                                           )

            labels, predictions = csqa_predictions(model, tokenizer, samples[:n_samples], trie)
            score = accuracy_score(y_true=labels, y_pred=predictions)
            population['eval'].append(score)

    elif task == "ContractNLI":
        population['f1_scores'] = []
        population['confusion_matrix'] = []
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):

            labels, predictions, per_doc_labels, per_doc_predictions = prompt_preds_contractnli_span(data_expanded[:n_samples], 
                                                                                                     task_description = prompts['task_description'][population['prompts'][i]['task_description']], 
                                                                                                     doc_description = prompts['doc_description'][population['prompts'][i]['doc_description']], 
                                                                                                     statement_description = prompts['statement_description'][population['prompts'][i]['statement_description']],
                                                                                                     answer_description = prompts['answer_description'][population['prompts'][i]['answer_description']],
                                                                                                     model = model,
                                                                                                     tokenizer = tokenizer,
                                                                                                     trie = trie,
                                                                                                     )
                                                                                                    
            #preds, n_not_founds = convert_preds_from_yesno_contractnli(predictions)
            preds, n_not_founds = convert_preds_from_yesno(predictions)
            score = accuracy_score(y_true=labels, y_pred=preds)
            print(f"score-->{score}")
            population['eval'].append(score)

            # f-1 score for more detailed analysis
            #f1_scores_per_class = f1_score(y_true=labels, y_pred=preds, average=None)

            f1s = []
            f1_c = 0
            f1_e = 0
            n_docs = len(per_doc_labels)
            print("PRINT")
            print(f"n_docs-->{n_docs}")
            n_examples=0
            for doc in range(n_docs):
                doc_preds = per_doc_predictions[doc]
                doc_labels = per_doc_labels[doc]
                n_examples+=len(doc_labels)
                #report = classification_report(y_true=labels, y_pred=preds)
                f1_scores_per_class_per_doc = f1_score(y_true=doc_labels, y_pred=doc_preds, average=None, labels=['Contradiction', 'Entailment'])
                print(f"f1_scores_per_class_per_doc-->{f1_scores_per_class_per_doc}")
                f1_c += f1_scores_per_class_per_doc[0]
                print(f"f1_c-->{f1_c}")
                f1_e += f1_scores_per_class_per_doc[1]
                print(f"f1_e-->{f1_e}")
            
            f1_c = f1_c/n_docs
            f1_e = f1_e/n_docs
            f1_scores_per_class = [f1_c, f1_e]
            print(f"f1_scores_per_class-->{f1_scores_per_class}")

            print(f"n_examples-->{n_examples}")
            print(f"len(predictions)-->{len(predictions)}")

            # Find unique class labels
            unique_labels = np.unique(np.concatenate((labels, preds)))
            # Pair each unique class label with its F1-score
            label_to_f1 = dict(zip(unique_labels, f1_scores_per_class))
            population['f1_scores'].append(label_to_f1)
            # confusion matrix
            cm = confusion_matrix(y_true=labels, y_pred=preds, labels=unique_labels)
            population['confusion_matrix'].append(cm)

    elif task == "MEDIQASUM":
        population['full_eval'] = []
        tt = 0
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):
            tt+=1
            #print(f"ttt 1--->{tt}")
            labels, predictions = prompt_preds_mediqasum(data_expanded[:n_samples], 
                                                         task_description = prompts['task_description'][population['prompts'][i]['task_description']], 
                                                         example_description = prompts['example_description'][population['prompts'][i]['example_description']], 
                                                         dialog_description = prompts['dialog_description'][population['prompts'][i]['dialog_description']],
                                                         answer_description =  prompts['answer_description'][population['prompts'][i]['answer_description']],
                                                         model=model,
                                                         tokenizer=tokenizer,
                                                         )
            
            #print(f"antes da eval{tt}")
            score = mediqasum_evaluation.evaluate_texts(predictions, labels, only_rouge = only_rouge)
            #print(f"depois da eval{tt}")
            population['eval'].append(score)
            #print(f"1111111 da eval{tt}")
            population['full_eval'].append(score)
            #print(f"2222222 da eval{tt}")

    elif task == "hyper_mutation":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):

            # semeval prompts
            semeval_prompts = extract_lines_to_dict('INITIAL_PROMPTS/SemEval_initial_population_prompts', task='SemEval')
            task_desc = semeval_prompts['task_description'][0]
            ctr_desc = semeval_prompts['ctr_description'][0]
            stat_desc = semeval_prompts['statement_description'][0]
            ans_desc = semeval_prompts['answer_description'][0]
            prompt = [task_desc, ctr_desc, stat_desc, ans_desc]

            #N = 6
            #mutation_prob = 0.8
            # loop to perform N mutations at random to a random number of parts of the base prompt
            # that's always the same
            # the score is then the averaged value 
            Scores_total = 0
            for _ in range(N):
                new_P = []
                for p in prompt:
                    if random.random() <= mutation_prob:
                        mutated = mutate_prompt(p, 
                                    mutation_prompt = prompts['mutation_prompts'][population['prompts'][i]['mutation_prompts']],
                                    model=model,
                                    tokenizer=tokenizer)
                    else:
                        mutated = p
                    new_P.append(mutated)
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                new_P[0],
                                                new_P[1],
                                                new_P[2],
                                                new_P[3],
                                                model, tokenizer, trie)
                score = f1_score(y_true=labels, y_pred=predictions, average='macro')
                Scores_total += score
            
            avg_score = Scores_total/N

            #print(f"score at eval-->{score}")
            population['eval'].append(avg_score)
    
    elif task == "hyper_crossover":
        for i in tqdm(range(n_pop), desc = f"Evaluating prompt population"):

            # semeval prompts
            semeval_prompts = extract_lines_to_dict('INITIAL_PROMPTS/SemEval_initial_population_prompts', task='SemEval')
            task_desc = semeval_prompts['task_description'][0]
            ctr_desc = semeval_prompts['ctr_description'][0]
            stat_desc = semeval_prompts['statement_description'][0]
            ans_desc = semeval_prompts['answer_description'][0]
            prompt_1 = [task_desc, ctr_desc, stat_desc, ans_desc]

            task_desc = semeval_prompts['task_description'][1]
            ctr_desc = semeval_prompts['ctr_description'][1]
            stat_desc = semeval_prompts['statement_description'][1]
            ans_desc = semeval_prompts['answer_description'][1]
            prompt_2 = [task_desc, ctr_desc, stat_desc, ans_desc]

            #N = 6
            #mutation_prob = 0.8
            # loop to perform N mutations at random to a random number of parts of the base prompt
            # that's always the same
            # the score is then the averaged value 
            Scores_total = 0
            for _ in range(N):
                new_P = []
                for p_1, p_2 in zip(prompt_1, prompt_2):
                    if random.random() <= mutation_prob:
                        crosovered = crossover_prompts(p_1,
                                                       p_2,
                                                       combination_prompt=prompts['combination_prompts'][population['prompts'][i]['combination_prompts']],
                                                       model=model,
                                                       tokenizer=tokenizer)

                    else:
                        crosovered = p_1
                    new_P.append(crosovered)
                labels, predictions = prompt_preds_semeval(data_expanded[:n_samples], 
                                                new_P[0],
                                                new_P[1],
                                                new_P[2],
                                                new_P[3],
                                                model, tokenizer, trie)
                score = f1_score(y_true=labels, y_pred=predictions, average='macro')
                Scores_total += score
            
            avg_score = Scores_total/N

            #print(f"score at eval-->{score}")
            population['eval'].append(avg_score)

    #print(f"at end of eval_po_function population['f1_scores']-->{population['f1_scores']}")
    return population

# create folder to store each run of the evo_alg function
def create_root_folder(task, 
                       alg = 'alg_2',
                       crossover_prob = 'nd',
                       mutation_prob = 'nd',
                       N = 'nd',
                       sampling_T = 'nd',
                       task_w_self_reasoning = 'nd',
                       task_w_highlight = 'nd'
                       ):
    # Format: Runs_YYYY-MM-DD_HH-MM-SS
    if alg=='alg_2':
        folder_name = datetime.now().strftime(f"RUNS_fine_tuning/{task}_whigh{task_w_highlight}_wself{task_w_self_reasoning}/Runs_%Y-%m-%d_%H-%M-%S_N{N}_cp{crossover_prob}_mp{mutation_prob}_sampT{sampling_T}")
    elif 'hyper' in task:
        folder_name = datetime.now().strftime(f"RUNS/{task}/Runs_%Y-%m-%d_%H-%M-%S_N{N}_cp{crossover_prob}_mp{mutation_prob}_sampT{sampling_T}_whigh{task_w_highlight}_wself{task_w_self_reasoning}")
    elif alg=='baseline':
        folder_name = datetime.now().strftime(f"RUNS/baseline/{task}/Runs_%Y-%m-%d_%H-%M-%S_N{N}_cp{crossover_prob}_mp{mutation_prob}_sampT{sampling_T}_whigh{task_w_highlight}_wself{task_w_self_reasoning}")
    else:
        folder_name = datetime.now().strftime(f"RUNS_{alg}_{task}/Runs_%Y-%m-%d_%H-%M-%S_N{N}_cp{crossover_prob}_mp{mutation_prob}_sampT{sampling_T}_whigh{task_w_highlight}_wself{task_w_self_reasoning}")

    os.makedirs(folder_name, exist_ok=True)
    return folder_name

# saving population at iteration
def save_population(iteration, population, root_folder, keep_list):
    # Create a folder for the current iteration
    iteration_folder = os.path.join(root_folder, f"Iteration_{iteration}")
    os.makedirs(iteration_folder, exist_ok=True)
    
    # Save each key in the population dictionary as a .txt file
    for key, values in population['prompts_dict'].items():
        file_path = os.path.join(iteration_folder, f"{key}.txt")
        with open(file_path, 'w') as file:
            i=0 
            for value in values:
                file.write(f"{i}->{value}\n")
                file.write("----------\n")  # Optional separator line
                i+=1
    
    # Save history
    for key, values in population['history'].items():
        file_path = os.path.join(iteration_folder, f"history_{key}.txt")
        with open(file_path, 'w') as file:
            i=0
            for value in values:
                file.write(f"{i}->{value}\n")
                file.write("----------\n")  # Optional separator line
                i+=1

    # Save the additional list in a separate .txt file
    additional_file_path = os.path.join(iteration_folder, "evaluations.txt")
    with open(additional_file_path, 'w') as file:
        for item in population['eval']:
            file.write(f"{item}\n")

    # save more metrics for contractnli task to trakc possible problem
    if population['task'] == 'ContractNLI' or population['task'] == 'SemEval':
        # Save the additional list in a separate .txt file
        additional_file_path = os.path.join(iteration_folder, "f1_scores.txt")
        with open(additional_file_path, 'w') as file:
            for item in population['f1_scores']:
                file.write(f"{item}\n")

        additional_file_path = os.path.join(iteration_folder, "confusion_matrix.txt")
        with open(additional_file_path, 'w') as file:
            for item in population['confusion_matrix']:
                file.write(f"{item}\n")

    # save more metrics for contractnli task to trakc possible problem
    if population['task'] == 'MEDIQASUM':
        # Save the additional list in a separate .txt file
        additional_file_path = os.path.join(iteration_folder, "full_eval.txt")
        with open(additional_file_path, 'w') as file:
            for item in population['full_eval']:
                file.write(f"{item}\n")
    
    # Save population in new format
    additional_file_path = os.path.join(iteration_folder, "population.txt")
    with open(additional_file_path, 'w') as file:
        for i in range(len(population['prompts'])):
            file.write(f"{population['prompts'][i]}, {population['eval'][i]}\n")

    # Save the additional list in a separate .txt file
    additional_file_path = os.path.join(iteration_folder, "keep_list.txt")
    with open(additional_file_path, 'w') as file:
        for item in keep_list:
            file.write(f"{item}\n")

    return None

def save_details(root_folder, n_pop, n_keep,
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

        file.write(f"Initial population size (size of suprompts pop): {n_pop}\n")
        file.write(f"Population size that's kept for next iteration: {n_keep}\n")
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

        file.write(f"Now the individuals are not bonded, meaning that new subprompts are created and randomly combined \n")
        if task == 'ContractNLI':
            file.write(f"ORACLE SPANS \n")

    # Save best f1 score at each iteration
    additional_file_path = os.path.join(root_folder, "scores_evo.txt")
    with open(additional_file_path, 'w') as file:
        for item in best_score_iterations:
            file.write(f"{item}\n")

    return None


def save_details_alg_2(root_folder, n_pop,
                 n_top, 
                 start_time, 
                 end_time,
                 patience,
                 max_iter,
                 iter,
                 best_score_iterations,
                 eval_data,
                 data_size,
                 task,
                 model_name,
                 quantize_model_4bits,
                 mutation_prob=0.5,
                 crossover_prob=0.5,
                 alg='alg_2',
                 N=None,
                 eval_mutation_prob = None,
                 best_iter=None,
                 evaluation_task = None,
                 retrieve_examples=None):
    
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

        file.write(f"Initial population size (size of suprompts pop): {n_pop}\n")

        file.write(f"How many of the top performers are being kept(elite): {n_top}\n")

        file.write(f"Max no. of iterations allowed: {max_iter}\n")
        file.write(f"Patience: {patience}\n")

        file.write(f"Evaluation done on: {eval_data} set\n")
        file.write(f"With {data_size} examples\n\n")

        file.write(f"Name of the model used: {model_name} \n")
        file.write(f"4 bit quantization: {quantize_model_4bits} \n")

        file.write(f"Now the individuals are not bonded, meaning that new subprompts are created and randomly combined \n")
        file.write(f"Retrieving examples-->{retrieve_examples} \n")
        
        if task == 'ContractNLI':
            file.write(f"ORACLE SPANS \n")

        if alg == 'alg_2':
            file.write(f"Alg 2 - Random sample 2 individuals weighted on scores, apply crossover and mutation to the offspring, top {n_top} performers are always kept as elite population\n")
            file.write(f"With crossover probability {crossover_prob} and mutation probability {mutation_prob}\n")
        elif 'hyper' in alg:
            file.write(f"Mutation evaluation done with N={N}, with probability={eval_mutation_prob}\n")
            file.write(f"Evaluation task - > {evaluation_task}\n")
        elif 'baseline' in alg:
            file.write(f"Mutation evaluation done with N={N}, with probability={eval_mutation_prob}\n")
            file.write(f"best result in baseline at iteration: {best_iter}\n")
    # Save best f1 score at each iteration
    additional_file_path = os.path.join(root_folder, "scores_evo.txt")
    with open(additional_file_path, 'w') as file:
        for item in best_score_iterations:
            file.write(f"{item}\n")

    return None


# function to sort prompt population based on evaluation values
def sort_pop(pop):
    population = deepcopy(pop)
    #print(f"population.keys()-->{population.keys()}")
    prompts = population['prompts']
    eval = population['eval']
    
    sorted_indices = np.argsort(eval)[:][::-1] # this is inverting the sorting order, so that element 0 is the largest
    sorted_prompts = [prompts[i] for i in sorted_indices if i < len(prompts)]
    sorted_eval = [eval[i] for i in sorted_indices if i < len(eval)]

    if population['task'] == 'ContractNLI' or population['task'] == 'SemEval':
        f1_scores = population['f1_scores']
        confusion_matrix = population['confusion_matrix']

        sorted_f1_scores = [f1_scores[i] for i in sorted_indices if i < len(f1_scores)]
        sorted_confusion_matrix = [confusion_matrix[i] for i in sorted_indices if i < len(confusion_matrix)]

        population['f1_scores'] = sorted_f1_scores
        population['confusion_matrix'] = sorted_confusion_matrix

    if population['task'] == 'MEDIQASUM':
        full_scores = population['full_eval']
        sorted_full_scores = [full_scores[i] for i in sorted_indices if i < len(full_scores)]
        population['full_eval'] = sorted_full_scores

    population['prompts'] = sorted_prompts
    population['eval'] = sorted_eval

    return population

# function to select population and respectives evaluations in an exploratory or exploitanional way
# n_pop is the number of individuals we want to keep
# n_top is the number of best performing individuals we want to keep, while the rest will be randomized
# if n_pop=n_top then full greedy search is done (only top candidates are kept
def pop_selection(population, # population (dictionary with keys: prompts and eval)
                  n_pop, # population size to be kept
                  n_top, # no. of top candidates to select (rest are randomized)
                  ):
    
    # because problems sometimes
    pop = deepcopy(population)

    sorted_pop = sort_pop(pop)

    sorted_prompts = sorted_pop['prompts']
    sorted_eval = sorted_pop['eval']
    
    keep_list = list(range(n_top)) + random.sample(range(n_top, len(pop['eval'])), k=n_pop-n_top)
    keep_list.sort()
    #print(f"keep_list-->{keep_list}")

    keep_prompts = [sorted_prompts[i] for i in keep_list]
    keep_eval = [sorted_eval[i] for i in keep_list]

    if pop['task'] == 'ContractNLI' or population['task'] == 'SemEval':
        sorted_f1_scores = sorted_pop['f1_scores']
        sorted_confusion_matrix = sorted_pop['confusion_matrix']

        keep_f1_scores = [sorted_f1_scores[i] for i in keep_list]
        keep_confusion_matrix = [sorted_confusion_matrix[i] for i in keep_list]

        pop['f1_scores'] = keep_f1_scores
        pop['confusion_matrix'] = keep_confusion_matrix
        #print(f"pop['f1_scores']-->{pop['f1_scores']}")
        #print(f"pop['confusion_matrix']-->{pop['confusion_matrix']}")
    
    if pop['task'] == 'MEDIQASUM':

        sorted_full_scores = sorted_pop['full_eval']
        keep_full_scores = [sorted_full_scores[i] for i in keep_list]
        pop['full_eval'] = keep_full_scores

    pop['prompts'] = keep_prompts
    pop['eval'] = keep_eval

    return update_population_and_prompts(pop), keep_list

# the number of individuals to be created will be determined by the number of subprompts
# dictionary with prompts
# initial flag just makes the order always the same
def create_population(task, prompts_dict, initial,
                      data_expanded, model, tokenizer, trie, n_samples,
                      n_pop = None,
                      history = None,
                      N=10, # for the hyper mutation thing
                      mutation_prob=0.8,
                      only_rouge = True,
                      save_preds4semeval_test = False,
                      folder = None,
                      task_w_one_shot = False,
                      task_w_highlight = False,
                      task_w_self_reasoning = False,
                      ):

    prompts = []
    population = []

    # check number of examples of each subprompts is the same
    tam = []
    #print(f"PROMPT_DICT-->{prompts_dict}")
    for key in prompts_dict:
        #print(len(prompts_dict[key]))
        tam.append(len(prompts_dict[key]))
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"!!!!!!!The no. of elements in each subprompt differs!!!!!!!")
        return None, 

    #print(f"prompts_dict-->{prompts_dict}")
    #print(f"tam-->{tam}")

    n_sub = tam[0]
    if n_pop == None:
        n_pop = n_sub
    
    #print(f"n_pop-->{n_pop}")
    
    if initial == True:
        for i in range(n_sub):
            p = {}
            for j in prompts_dict:
                p[j] = i
            prompts.append(p)
    else:
        indices = {}
        for i in prompts_dict:
            indices[i] = random.sample(list(range(n_sub)), n_sub)

        #print(f"indices-->{indices}")
        for i in range(n_sub):
            prompts_index = {}
            for j in prompts_dict:
                #print(f"indices[j,i]-->{indices[j,i]}")
                prompts_index[j] = indices[j][i]
            prompts.append(prompts_index)

    # if number of individuals larger then the number of subprompts then there will be repetitions allowed
    if n_pop > n_sub:
        indices = {}
        for i in prompts_dict:
            indices[i] = random.choices(list(range(n_sub)), k = n_pop-n_sub)

        #print(f"indices-->{indices}")
        for i in range(n_sub):
            prompts_index = {}
            for j in prompts_dict:
                #print(f"indices[j,i]-->{indices[j,i]}")
                prompts_index[j] = indices[j][i]
            prompts.append(prompts_index)

    if initial == True and history == None:
        history = deepcopy(prompts_dict)
        for key in history:
            history[key] = n_sub * ['initial']

    population = {'task': task, 'prompts_dict': prompts_dict, 'prompts': prompts, 'eval': [], 'history': history}

    # evaluate population that was created
    #print(f"antes de criar")
    population = eval_pop(population, data_expanded = data_expanded, 
                            model=model, tokenizer=tokenizer, trie=trie, n_samples = n_samples, N=N, mutation_prob=mutation_prob,
                            only_rouge=only_rouge,
                            save_preds4semeval_test = save_preds4semeval_test, folder = folder,
                            task_w_one_shot = task_w_one_shot,
                            task_w_self_reasoning = task_w_self_reasoning,
                            task_w_highlight = task_w_highlight,)

    #print(f"depois de criar")
    return population

# function to combine two populations
# assigning new indices to match the new order
def combine_populations(pop_1, pop_2):
    combined_pop = deepcopy(pop_1)
    
    combined_pop['eval'] += pop_2['eval']

    if pop_1['task'] == 'ContractNLI' or pop_1['task'] == 'SemEval':
        combined_pop['f1_scores'] += pop_2['f1_scores']
        combined_pop['confusion_matrix'] += pop_2['confusion_matrix']
    
    if pop_1['task'] == 'MEDIQASUM':
        combined_pop['full_eval'] += pop_2['full_eval']

    # Calculate the size of prompts_dict for each key to correctly adjust indices
    prompts_dict_sizes = {key: len(combined_pop['prompts_dict'][key]) for key in combined_pop['prompts_dict'].keys()}

    # Update indices of pop_2 and append to combined_pop['prompts']
    for i in range(len(pop_2['prompts'])):
        updated_indices = {}
        for key in combined_pop['prompts_dict']:
            updated_indices[key] = pop_2['prompts'][i][key] + prompts_dict_sizes[key]
        combined_pop['prompts'].append(updated_indices)
    
    # Concatenate the prompts_dict and history from pop_2 to combined_pop
    for key in combined_pop['prompts_dict'].keys():
        combined_pop['prompts_dict'][key] += pop_2['prompts_dict'][key]
        combined_pop['history'][key] += pop_2['history'][key]

    return combined_pop

# function to update the prompt_dict part, to remove the suprompts that are not selected for next gen
# while updating the indices in the prompts values in order to match the new order
# done after removal of the 'prompt' (meaning the indices part id'ind the individuals)
def update_population_and_prompts(population):
    # Step 1: Collect used indices for each key
    keys = population['prompts_dict'].keys()
    
    used_indices = {key: set() for key in keys}
    
    for sublist in population['prompts']:
        for key in sublist:
            used_indices[key].add(sublist[key])

    # Step 2: Create new index mappings and update population['prompts']
    index_mapping = {
        key: {old_index: new_index for new_index, old_index in enumerate(sorted(indices))}
        for key, indices in used_indices.items()
    }
    
    for sublist in population['prompts']:
        for key in sublist:
            sublist[key] = index_mapping[key][sublist[key]]

    # Step 3: Update the prompts_dict by removing unreferenced strings
    for key in keys:
        population['prompts_dict'][key] = [population['prompts_dict'][key][old_index] for old_index in sorted(used_indices[key])]
        population['history'][key] = [population['history'][key][old_index] for old_index in sorted(used_indices[key])]
    return population

# function to evaluate the best prompt obtained after the evolutionary algorithm
# tipycally it'll be in the best iter folder for a given run
# saves the results as a txt to that folder
def test_eval(task,
              RUN_folder_path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_preds4semeval_test=False, # is turned to true if it's the semeval task
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False
              ):


    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)

    if task == 'MEDIQASUM':
        trie = None
    else:
        trie = get_Marisa_Trie(task, tokenizer)
    
    best_path = os.path.join(RUN_folder_path, 'Iteration_best/')    
    best_prompts = extract_lines_to_dict(best_path, task=task)
    #print(f"best_prompts-->{best_prompts}")

    if task == 'SemEval' or task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = 'gold_test')
        save_preds4semeval_test = True
    #elif task == "CSQA":
        #data_expanded = extract_CSQA_data(type = eval_data)
    elif task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = 'test')
    elif task == "MEDIQASUM":
        data_expanded = extract_MEDIQASUM_data()

    # criar pop e avaliar
    best_pop = create_population(task, best_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples=0, only_rouge=False, 
                                           save_preds4semeval_test = save_preds4semeval_test,
                                           folder = RUN_folder_path+'/Iteration_best/')
    

    score = best_pop['eval']
    if 'SemEval' in task:
        semeval_test_evaluation(pred_filename=RUN_folder_path + '/Iteration_best/test_predictions.json',
                                gold_filename='DATASETS/SemEval_data/gold_test.json',
                                output_dir=RUN_folder_path + '/Iteration_best'
                                )
    
    # write to new txt in folder
    file_name = RUN_folder_path+'/Iteration_best/test_evaluation.txt'
    with open(file_name, 'w') as file:
        if 'SemEval' in task:
            f1_scores = best_pop['f1_scores']
            confusion_matrix = best_pop['confusion_matrix']
            file.write(f"Macro F1 score: {score}\n")
            file.write(f"F1 scores: {f1_scores}\n")
            file.write(f"Confusion matrix: {confusion_matrix}\n")
        elif task == "CSQA":
            file.write(f"Acc score: {score}\n")
        elif task == "ContractNLI":
            f1_scores = best_pop['f1_scores']
            confusion_matrix = best_pop['confusion_matrix']
            file.write(f"Acc score: {score}\n")
            file.write(f"F1 scores: {f1_scores}\n")
            file.write(f"Confusion matrix: {confusion_matrix}\n")
        elif task == 'MEDIQASUM':
            full_scores = best_pop['eval']
            file.write(f"Full scores: {full_scores}\n")

    print(f"test set evaluation-->{score}, saved to {file_name}")
    return None

# function to dillate scores so there's a bigger different between the worst and best performing individuals
# factor of eps added so to not have 0 probability for the lowest value
def min_max(scores, eps=0.05):
    scores = np.array(scores)
    new_scores = (scores - scores.min())/(scores.max() - scores.min())
    new_scores = new_scores/new_scores.sum()
    new_scores = new_scores + eps
    new_scores = new_scores/new_scores.sum()
    return new_scores

# for sampling with softmax
def softmax(x): return np.exp(x) / np.sum(np.exp(x), axis=0)


# !!!!!!! DEPRECATED since you updated the representation of population['prompt'] from a list of list to a lsit of dictionaries
# function to run the evolutionary alg, with a initial population of prompts
# evolutionary prompts (1 for mutation, 1 for combination)
# hf model and tokenizer
# hyperparameters of the algorithm
def evo_alg(task, initial_prompts, evolutionary_prompts,
            model_name = "mistralai/Mistral-7B-Instruct-v0.2",
            quantize_model_4bits = True,
            n_pop = 5, # initial population size and the number of elements kepts at each iteration
            n_keep = 5,
            n_top = 5, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
            n_combinations = 15,
            patience = 10,
            max_iter = 50,
            temperature = 1.0, #temperature for decoding combined and mutated
            top_p = 0.8, #sampling for decoding combined and mutated
            save = True,
            eval_data = 'dev', # dev or train
            data_size = 0): # no. of samples where the prompts are evaluated, if =0 all are used

    if task != 'SemEval' and task != 'SemEval_self' and task != 'CSQA' and task != 'ContractNLI':
        print(f"Incorrect task selected")
        return None
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
        
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == True:
        n_pop = tam[0]
    else:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # check n_top
    if n_top > n_pop:
        n_top = n_pop

    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(task, tokenizer)
    
    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task)
        print(f"Root folder created: {root_folder}")

    if task == 'SemEval' or task == 'SemEval_self':
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

    initial_population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size)

    print(f"initial_population eval-->{initial_population['eval']}")
    best_score_iterations.append(max(initial_population['eval']))
    
    if save == True:
        save_population('initial', initial_population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 
        mutated_prompts = {key: [] for key in initial_population['prompts_dict'].keys()}
        combined_prompts = {key: [] for key in initial_population['prompts_dict'].keys()}
        mutated_history = {key: [] for key in initial_population['prompts_dict'].keys()}
        combined_history = {key: [] for key in initial_population['prompts_dict'].keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop), desc = f"iteration {iter} - Mutating prompts"):
            # iterate through the subprompts
            for j in initial_population['prompts_dict'].keys():

                # mutate each subprompt and add to the mutated population prompts
                mutated = mutate_prompt(initial_population['prompts_dict'][j][i], evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 

                mutated_prompts[j].append(mutated)
                mutated_history[j].append(f"mutated from {i} at iteration {iter+1}")

        mutated_population = create_population(task, mutated_prompts, initial = False,
                                               data_expanded = data_expanded, 
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=mutated_history)

        #print(f"initial_population['eval']-->{initial_population['eval']}")
        population = combine_populations(initial_population, mutated_population)
        #print(f"population['eval']-->{population['eval']}")
        print(f"initial_population['eval']-->{initial_population['eval']}")

        for i in tqdm(range(n_combinations), desc = f"iteration {iter} - Combining prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            m = 0
            for j in population['prompts_dict'].keys():

                sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2)
                print(f"sel4comb-->{sel4comb}")

                # combine each subprompt randomly selected and add to the combined and total population
                combined = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][m]], population['prompts_dict'][j][population['prompts'][sel4comb[1]][m]],
                                           evolutionary_prompts['combination_prompts'][0], model, tokenizer)
                combined_prompts[j].append(combined)
                combined_history[j].append(f"combined from [{population['prompts'][sel4comb[0]][m]}] and [{population['prompts'][sel4comb[1]][m]}] at iteration {iter+1}")
                m+=1

        combined_population = create_population(task, combined_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history=combined_history)

        population = combine_populations(population, combined_population)

         # if improved patience returns to 0
        #print(f"before pat counter determination")
        #print(f"max(population['eval'])-->{max(population['eval'])}")
        #print(f"max(initial_population['eval'])-->{max(initial_population['eval'])}")
        #print(f"max(population['eval']) > max(initial_population['eval'])-->{max(population['eval']) > max(initial_population['eval'])}")
        if max(population['eval']) > max(initial_population['eval']):
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")

        keep_pop = deepcopy(sorted_population)

        # Create a new dictionary with the same keys, but values are lists with only the selected indices
        # how the population is being maintained
        n_pop = n_keep
        keep_pop, keep_list = pop_selection(keep_pop, n_pop, n_top) # !!!!!!!!!!

        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, keep_list)
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

        initial_population = deepcopy(keep_pop)
        print(f"evaluation of keepers for next gen-->{initial_population['eval']}")
        print(f"keep_list-->{keep_list}")

        print(f"patience_counter-->{patience_counter}")

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_pop, keep_list = pop_selection(sorted_population, 1, 1)
    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details(root_folder, n_pop, n_keep, 
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

    return best_pop, best_score_iterations




##############################################################
def evo_alg_2(task, 
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              n_pop = 5, # initial population size and the number of elements kepts at each iteration
              n_top = 1, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
              mutation_prob=0.5,
              crossover_prob=0.5,
              sampling_T = 5.0,
              patience = 20,
              max_iter = 200,
              save = True,
              eval_data = 'dev', # dev or train
              data_size = 0, # no. of samples where the prompts are evaluated, if =0 all are used
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              do_test_eval = True,
              ): 
    
    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    
    data_expanded, initial_prompts, evolutionary_prompts, trie = sel_task_dataset_initial_prompts_evo_prompts(task_name=task,
                                                                                                        tokenizer=tokenizer,
                                                                                                        w_one_shot=task_w_one_shot,
                                                                                                        w_self_reasoning=task_w_self_reasoning,
                                                                                                        w_highlight=task_w_highlight
                                                                                                        )

    print(f"initial_prompts.keys()-->{initial_prompts.keys()}")
    
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task,
                                         crossover_prob=crossover_prob,
                                         mutation_prob=mutation_prob,
                                         N=n_pop,
                                         sampling_T=sampling_T,
                                         task_w_self_reasoning = task_w_self_reasoning,
                                         task_w_highlight = task_w_highlight)
        print(f"Root folder created: {root_folder}")

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, 
                                   initial_prompts, 
                                   initial = True,
                                   n_pop=n_pop,
                                   data_expanded = data_expanded, 
                                   model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                   task_w_one_shot = task_w_one_shot,
                                   task_w_highlight = task_w_highlight,
                                   task_w_self_reasoning = task_w_self_reasoning)
    
    #print(f"AT START-->population['prompts']-->{population['prompts']}")
    #print(f"AT START-->population-->{population}")

    n_sub = len(population['prompts_dict'][list(population['prompts_dict'].keys())[0]])
    #print(f"n_sub-->{n_sub}")

    # increase intial population by generating new prompts
    """
    # if needed increase initial population by generating mutated variations of the initial promtp list
    if n_pop > tam[0]:
        print(f"n_pop > tam[0]-->{n_pop} > {tam[0]}")

        extra_prompts = {key: [] for key in initial_prompts.keys()}
        extra_history = {key: [] for key in initial_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop-tam[0]), desc = f"Generating extra initial pop"):
            # iterate through the subprompts
            for j in initial_prompts.keys():
                # mutate each subprompt and add to the mutated population prompts
                #rndm choice of the operator to be used
                mutation_prompt_index = random.choice(list(range(len(evolutionary_prompts['mutation_prompts']))))
                #random choice of the prompt to be mutated
                ind_prompt = random.choice(list(range(tam[0])))
                mutated = mutate_prompt(initial_prompts[j][ind_prompt], evolutionary_prompts['mutation_prompts'][mutation_prompt_index], 
                                        model, tokenizer) 
                
                hist = f"mutated from {ind_prompt}, using mutation prompt {mutation_prompt_index}"

                extra_prompts[j].append(mutated)
                extra_history[j].append(hist)

        extra_population = create_population(task, extra_prompts, initial = True,
                                               data_expanded = data_expanded,
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=extra_history, 
                                               task_w_one_shot = task_w_one_shot,
                                               task_w_highlight = task_w_highlight,
                                               task_w_self_reasoning = task_w_self_reasoning)
        population = combine_populations(population, extra_population)
    """
    patience_counter = 0
    iter = 0
    #print(f"initial_population eval-->{population['eval']}")
    best_score_iterations.append(max(population['eval']))
    
    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        best_score_at_start = max(population['eval'])
        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_sub-n_top), desc = f"iteration {iter} - generating off springs prompts"):
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():

                soft_max_scores = softmax(np.array(population['eval'])/sampling_T)
                #print(f"soft_max_scores-->{soft_max_scores}")
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=2, replace=False, p = soft_max_scores)) 
                #print(f"sel4comb-->{sel4comb}")

                # apply crossover with probability crossover_prob, else off spring is copy of parent
                if random.random() <= crossover_prob:
                    # combine each subprompt randomly selected and add to the combined and total population
                    combined = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]],
                                           evolutionary_prompts['combination_prompts'][0], model, tokenizer)
                    hist = f"crossover between [{population['prompts'][sel4comb[0]][j]}] and [{population['prompts'][sel4comb[1]][j]}]"
                else:
                    if population['eval'][sel4comb[0]] > population['eval'][sel4comb[1]]:
                        combined = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                        hist = f"copy of [{population['prompts'][sel4comb[0]][j]}]"
                    else:
                        combined = population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]]
                        hist = f"copy of [{population['prompts'][sel4comb[1]][j]}]"
                
                # apply mutation with probability crossover_prob, else off spring remains the same
                if random.random() <= mutation_prob:
                    mutated = mutate_prompt(combined, evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer) 
                    hist+=f" followed by mutation"
                else:
                    mutated = combined
                    hist+=f" "

                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(mutated)
                offspring_history[j].append(hist)

        offspring_population = create_population(task, offspring_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history = offspring_history, 
                                                task_w_one_shot = task_w_one_shot,
                                                task_w_highlight = task_w_highlight,
                                                task_w_self_reasoning = task_w_self_reasoning,
                                                n_pop=n_pop-n_top,)

        #print(f"elite_population['prompts']-->{elite_population['prompts']}")
        #print(f"offspring_population['prompts']-->{offspring_population['prompts']}")
        #print(f"AT MID-->elite_population-->{elite_population}")
        #print(f"AT MID-->offspring_population-->{offspring_population}")
        if n_top ==0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)
        
        #print(f"AFTER COMBINATION-->population['prompts']-->{population['prompts']}")
        #print(f"AFTER COMBINATION-->population-->{population}")

        if max(population['eval']) > best_score_at_start:
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        #print(f"population['prompts']-->{population['prompts']}")
        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")


        # Call the function
        if save == True:
            #print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_pop, keep_list = pop_selection(sorted_population, 1, 1)
    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            mutation_prob,
                            crossover_prob,
                            retrieve_examples=retrieve_examples,
                            alg='alg_2',
                            )
            
            create_plots_from_RUNS_folder(root_folder)

    if do_test_eval == True:
        """try:
            print(f"test set evaluation")
            test_eval(task=task, RUN_folder_path = root_folder, model_name=model_name)
        except:
            print(f"TEST FINAL EVALUATION FAILED")"""
        print(f"test set evaluation")
        test_eval(task=task, RUN_folder_path = root_folder, model_name=model_name)

    return best_pop, best_score_iterations


##############################################################
# baseline algorithm that performs a MC search
# only applies 
def alg_baseline(task, initial_prompts, evolutionary_prompts,
            model_name = "mistralai/Mistral-7B-Instruct-v0.2",
            quantize_model_4bits = True,
            n_pop = 5, # initial population size and the number of elements kepts at each iteration
            n_top = 0, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
            mutation_prob=0.5,
            patience = 10,
            max_iter = 50,
            temperature = 1.0, #temperature for decoding combined and mutated
            top_p = 0.8, #sampling for decoding combined and mutated
            save = True,
            eval_data = 'dev', # dev or train
            data_size = 0,
            test_eval = True ): # no. of samples where the prompts are evaluated, if =0 all are used

    if task != 'SemEval' and task != 'SemEval_self' and task != 'CSQA' and task != 'ContractNLI':
        print(f"Incorrect task selected")
        return None
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(task, tokenizer)

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()

    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task, alg = 'baseline')
        print(f"Root folder created: {root_folder}")

    if task == 'SemEval' or task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = eval_data)

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                           n_pop=n_pop,)

    # if needed increase initial population by generating mutated variations of the initial promtp list
    if n_pop > tam[0]:
        print(f"n_pop > tam[0]-->{n_pop} > {tam[0]}")

        extra_prompts = {key: [] for key in initial_prompts.keys()}
        extra_history = {key: [] for key in initial_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop-tam[0]), desc = f"Generating extra initial pop"):
            # iterate through the subprompts
            for j in initial_prompts.keys():
                # mutate each subprompt and add to the mutated population prompts
                #rndm choice of the operator to be used
                mutation_prompt_index = random.choice(list(range(len(evolutionary_prompts['mutation_prompts']))))
                #random choice of the prompt to be mutated
                ind_prompt = random.choice(list(range(tam[0])))
                mutated = mutate_prompt(initial_prompts[j][ind_prompt], evolutionary_prompts['mutation_prompts'][mutation_prompt_index], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                
                hist = f"mutated from {ind_prompt}, using mutation prompt {mutation_prompt_index}"

                extra_prompts[j].append(mutated)
                extra_history[j].append(hist)

        extra_population = create_population(task, extra_prompts, initial = True,
                                               data_expanded = data_expanded,
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=extra_history,
                                               n_pop=n_pop,)
        population = combine_populations(population, extra_population)

    patience_counter = 0
    iter = 0
    print(f"initial_population eval-->{population['eval']}")
    #print(f"len(population['eval'])-->{len(population['eval'])}")
    best_score_iterations.append(max(population['eval']))

    # pop with best individual at start
    best_pop, _ = pop_selection(population, 1, 1)
    best_iter = iter

    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_pop-n_top), desc = f"iteration {iter} - generating off springs prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():

                # random choice (NOT weighted)
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=1, replace=False)) #better way
                #print(f"sel4comb-->{sel4comb}")

                # apply mutation with probability crossover_prob, else off spring remains the same
                if random.random() <= mutation_prob:
                    mutated = mutate_prompt(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                    hist = f"mutation of [{population['prompts'][sel4comb[0]][j]}]"
                else:
                    mutated = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                    hist=f" copy of [{population['prompts'][sel4comb[0]][j]}]"

                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(mutated)
                offspring_history[j].append(hist)

        offspring_population = create_population(task, offspring_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history = offspring_history,
                                                n_pop=n_pop,)

        if n_top == 0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)


        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")

        # upgraded best one that is stored if improvement is shown
        new_best_pop, _ = pop_selection(sorted_population, 1, 1)
        if new_best_pop['eval'][0] > best_pop['eval'][0]:
            best_pop = deepcopy(new_best_pop)
            best_iter = iter+1
            patience_counter = 0
        else:
            patience_counter += 1

        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1


    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            alg='baseline',
                            best_iter=best_iter)
            
            create_plots_from_RUNS_folder(root_folder)

    if test_eval == True:
        try:
            print(f"test set evaluation")
            test_eval(task=task, RUN_folder_path = root_folder, model_name=model_name)
        except:
            print('error in the test set predictions')
            pass

    return best_pop, best_score_iterations


##############################################################

# hyperevolution algorith
# mutation pormpts are evaluatedd by generating N variations for a given set of subprompts
# the score will be the average across those N examples
def evo_alg_hyper(task,
                evaluation_task,
                initial_prompts, 
                hyper_evolutionary_prompts,
                model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                quantize_model_4bits = True,
                n_pop = 5, # initial population size and the number of elements kepts at each iteration
                n_top = 1, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                crossover_prob=0.5,
                sampling_T = 5.0,
                patience = 10,
                max_iter = 50,
                temperature = 1.0, #temperature for decoding combined and mutated
                top_p = 0.8, #sampling for decoding combined and mutated
                save = True,
                eval_data = 'dev', # dev or train
                data_size = 0, # no. of samples where the prompts are evaluated, if =0 all are used
                N = 10,
                eval_mutation_prob = 0.8):
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # load model and tokenizer
    # wether or not to quantize model

    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(evaluation_task, tokenizer)

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task, alg=task)
        print(f"Root folder created: {root_folder}")

    if evaluation_task == 'SemEval' or evaluation_task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif evaluation_task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif evaluation_task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = eval_data)

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                           N=N,
                                           mutation_prob=eval_mutation_prob)

    # if needed increase initial population by generating mutated variations of the initial promtp list
    if n_pop > tam[0]:
        print(f"n_pop > tam[0]-->{n_pop} > {tam[0]}")

        extra_prompts = {key: [] for key in initial_prompts.keys()}
        extra_history = {key: [] for key in initial_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop-tam[0]), desc = f"Generating extra initial pop"):
            # iterate through the subprompts
            for j in initial_prompts.keys():
                # mutate each subprompt and add to the mutated population prompts
                #rndm choice of the operator to be used
                mutation_prompt_index = random.choice(list(range(len(hyper_evolutionary_prompts['mutation_prompts']))))
                #random choice of the prompt to be mutated
                ind_prompt = random.choice(list(range(tam[0])))
                mutated = mutate_prompt(initial_prompts[j][ind_prompt], hyper_evolutionary_prompts['mutation_prompts'][mutation_prompt_index], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                
                hist = f"mutated from {ind_prompt}, using hyper-mutation prompt {mutation_prompt_index}"

                extra_prompts[j].append(mutated)
                extra_history[j].append(hist)

        extra_population = create_population(task, extra_prompts, initial = True,
                                               data_expanded = data_expanded,
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=extra_history,
                                               N=N,
                                               mutation_prob=eval_mutation_prob)
        
        population = combine_populations(population, extra_population)

    patience_counter = 0
    iter = 0
    print(f"initial_population eval-->{population['eval']}")
    #print(f"len(population['eval'])-->{len(population['eval'])}")
    best_score_iterations.append(max(population['eval']))
    
    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        best_score_at_start = max(population['eval'])
        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_pop-n_top), desc = f"iteration {iter} - generating off springs prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():
                #print(f"len(population['eval'])-->{len(population['eval'])}")
                soft_max_scores = softmax(np.array(population['eval'])/sampling_T)
                #print(f"soft_max_scores-->{soft_max_scores}")
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=2, replace=False, p = soft_max_scores)) #better way

                # apply crossover with probability crossover_prob, else off spring is copy of parent
                if random.random() <= crossover_prob:
                    # combine each subprompt randomly selected and add to the combined and total population
                    combined = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]],
                                           hyper_evolutionary_prompts['combination_prompts'][0], model, tokenizer,
                                           temperature=temperature, top_p=top_p)
                    hist = f"crossover between [{population['prompts'][sel4comb[0]][j]}] and [{population['prompts'][sel4comb[1]][j]}]"
                else:
                    combined = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                    hist = f"copy of [{population['prompts'][sel4comb[0]][j]}]"
                
                # apply mutation with probability crossover_prob, else off spring remains the same
                if random.random() <= mutation_prob:
                    mutated = mutate_prompt(combined, hyper_evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                    hist+=f" followed by mutation"
                else:
                    mutated = combined
                    hist+=f" "

                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(mutated)
                offspring_history[j].append(hist)

        offspring_population = create_population(task, offspring_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history = offspring_history,
                                                N=N,
                                                mutation_prob=eval_mutation_prob)

        if n_top ==0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)

        if max(population['eval']) > best_score_at_start:
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")


        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_pop, keep_list = pop_selection(sorted_population, 1, 1)
    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            alg=task,
                            N=N,
                            eval_mutation_prob=eval_mutation_prob,
                            evaluation_task=evaluation_task,)
            
            create_plots_from_RUNS_folder(root_folder)

    return best_pop, best_score_iterations

##############################################################


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
    if "SemEval" in directory_path or "hyper" in directory_path:
        ymin = 0.50
        ymax = 0.75
        score = 'F1-Score'
    elif 'CSQA' in directory_path:
        ymin = 0.50
        ymax = 0.80
        score = 'Accuracy'
    elif 'ContractNLI' in directory_path:
        ymin = 0.70
        ymax = 0.95
        score = 'Accuracy'
    elif 'MEDIQASUM' in directory_path:
        ymin = 0
        ymax = 100
        score = 'Rouge-1 F1'
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
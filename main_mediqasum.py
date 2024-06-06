import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

import evo_functions as evo

folder_path = 'INITIAL_PROMPTS/MEDIQASUM_initial_population_prompts'
initial_population_prompts = evo.extract_lines_to_dict(folder_path, task = "MEDIQASUM")

evolutionary_prompts = evo.extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "Evo_prompts")

best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", initial_prompts = initial_population_prompts, 
                                                  evolutionary_prompts = evolutionary_prompts, 
                                                  #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                  model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                  quantize_model_4bits = False,
                                                  n_pop = 5,
                                                  n_top = 1,
                                                  mutation_prob=0.5,
                                                  crossover_prob=0.5,
                                                  patience = 2,
                                                  max_iter = 3,
                                                  data_size = 0) # number of examples where the prompts are evaluated
                                                                # 0 means all"""


"""from evo_functions import extract_lines_to_dict, prompt_preds_mediqasum, extract_MEDIQASUM_data, load_model
from mediqasum_evaluation import evaluate_texts

import csv
import os
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

a = extract_lines_to_dict('MEDIQASUM_initial_population_prompts', 'MEDIQASUM')
print(f'a.keys()-->{a.keys()}')

task_description = a['task_description'][0]
example_description = a['example_description'][0]
dialog_description = a['dialog_description'][0]
answer_description = a['answer_description'][0]

b = extract_MEDIQASUM_data('MEDIQASUM_data')
#print(f'b[0]-->{b[0]}')
print(f'b[0].keys()-->{b[0].keys()}')

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = load_model(checkpoint = model_name, quantized = True)

labels, predictions = prompt_preds_mediqasum(b[:], task_description, example_description, dialog_description, answer_description, model, tokenizer)

import nltk
nltk.download('punkt')

del model, tokenizer
torch.cuda.empty_cache()

print()
evaluate_texts(predictions, labels)


"""
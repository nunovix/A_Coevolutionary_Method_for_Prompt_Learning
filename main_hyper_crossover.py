import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

from evo_functions import extract_lines_to_dict, evo_alg_hyper

crossover_prompts = extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "hyper_crossover")
#print(f"mutation_prompts-->{mutation_prompts}")

hyper_evolutionary_prompts = extract_lines_to_dict("INITIAL_PROMPTS/hyper_evolutionary_prompts", task = "Evo_prompts")
#print(f"hyper_evolutionary_prompts-->{hyper_evolutionary_prompts}")

evo_alg_hyper(task = 'hyper_crossover',
                evaluation_task = 'SemEval',
                initial_prompts = crossover_prompts, 
                hyper_evolutionary_prompts = hyper_evolutionary_prompts,
                #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                model_name = "microsoft/Phi-3-mini-128k-instruct",
                quantize_model_4bits = True,
                n_pop = 5, # initial population size and the number of elements kepts at each iteration
                n_top = 1, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                crossover_prob=0.5,
                patience = 10,
                max_iter = 200,
                data_size = 0,
                N=10,
                eval_mutation_prob = 0.8)
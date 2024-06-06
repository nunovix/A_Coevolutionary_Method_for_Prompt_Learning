import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo


best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                   model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                   quantize_model_4bits = True,
                                                   patience = 2,
                                                   max_iter = 4,
                                                   data_size = 20, # number of examples where the prompts are evaluate 0 means all
                                                   task_w_one_shot = False,
                                                   task_w_self_reasoning = False,
                                                   task_w_highlight = False,
                                                   do_test_eval = False
                                                   ) #"""

























"""
import evo_functions as evo
import os

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

folder_path = 'INITIAL_PROMPTS/ContractNLI_initial_population_prompts'
small_folder_path = 'INITIAL_PROMPTS/small_ContractNLI_initial_population_prompts'

initial_population_prompts = evo.extract_lines_to_dict(folder_path, task = "ContractNLI")
small_initial_population_prompts = evo.extract_lines_to_dict(small_folder_path, task = "ContractNLI")

evolutionary_prompts = evo.extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "Evo_prompts")

best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", initial_prompts = initial_population_prompts, 
                                                  evolutionary_prompts = evolutionary_prompts, 
                                                  model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                  quantize_model_4bits = True,
                                                  n_pop = 5,
                                                  n_top = 1,
                                                  mutation_prob=0.5,
                                                  crossover_prob=0.5,
                                                  patience = 20,
                                                  max_iter = 200,
                                                  data_size = 0) # number of examples where the prompts are evaluated
                                                                # 0 means all"""


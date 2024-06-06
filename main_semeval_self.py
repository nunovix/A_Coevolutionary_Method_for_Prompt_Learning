import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import evo_functions as evo

folder_path = 'INITIAL_PROMPTS/SemEval_self_initial_population_prompts'
#small_folder_path = 'small_SemEval_initial_population_prompts'

initial_population_prompts = evo.extract_lines_to_dict(folder_path, task = "SemEval_self")
#small_initial_population_prompts = evo.extract_lines_to_dict(small_folder_path)

evolutionary_prompts = evo.extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "Evo_prompts")


best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval_self", 
                                                  #initial_prompts = initial_population_prompts, 
                                                  #evolutionary_prompts = evolutionary_prompts, 
                                                  #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                  model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                  quantize_model_4bits = True,
                                                  n_pop = 5,
                                                  n_top = 1,
                                                  mutation_prob=0.5,
                                                  crossover_prob=0.5,
                                                  patience = 15,
                                                  max_iter = 200,
                                                  data_size = 0) # number of examples where the prompts are evaluated
                                                                # 0 means all"""

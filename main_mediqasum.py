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
                                                  model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                  quantize_model_4bits = False,
                                                  n_pop = 5,
                                                  n_top = 1,
                                                  mutation_prob=0.5,
                                                  crossover_prob=0.5,
                                                  patience = 2,
                                                  max_iter = 3,
                                                  data_size = 0) # number of examples where the prompts are evaluated
                                                                # 0 means all"""


best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_highlight = True,
                                                            task_w_oracle_spans = True,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            )
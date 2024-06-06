import evo_functions as evo
import os

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

folder_path = 'INITIAL_PROMPTS/CSQA_initial_population_prompts'
small_folder_path = 'INITIAL_PROMPTS/small_CSQA_initial_population_prompts'

initial_population_prompts = evo.extract_lines_to_dict(folder_path)
small_initial_population_prompts = evo.extract_lines_to_dict(small_folder_path)

evolutionary_prompts = evo.extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts")

"""best_prompt, best_score_iterations = evo.evo_alg(task = "CSQA", initial_population_prompts = initial_population_prompts, 
                                                  evolutionary_prompts = evolutionary_prompts,
                                                  model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                  quantize_model_4bits = True,
                                                  n_pop = 5,
                                                  n_top = 2,
                                                  n_combinations = 3,
                                                  patience = 2,
                                                  max_iter = 5,
                                                  temperature=1.0, #mutations and combinations
                                                  top_p=0.8,
                                                  data_size = 100) # number of examples where the prompts are evaluated"""

best_prompt, best_score_iterations = evo.evo_alg(task = "CSQA", initial_population_prompts = initial_population_prompts, 
                                                  evolutionary_prompts = evolutionary_prompts,
                                                  model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                  quantize_model_4bits = True,
                                                  n_pop = 5,
                                                  n_top = 2,
                                                  n_combinations = 15,
                                                  patience = 10,
                                                  max_iter = 200,
                                                  temperature=1.0, #mutations and combinations
                                                  top_p=0.8,
                                                  data_size = 0) # number of examples where the prompts are evaluated
                                                                # 0 means all

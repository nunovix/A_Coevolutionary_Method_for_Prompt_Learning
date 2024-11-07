import evo_functions as evo
import os

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
folder_path = 'INITIAL_PROMPTS/SemEval_initial_population_prompts'
small_folder_path = 'INITIAL_PROMPTS/small_SemEval_initial_population_prompts'

initial_population_prompts = evo.extract_lines_to_dict(folder_path, task = "SemEval")
small_initial_population_prompts = evo.extract_lines_to_dict(small_folder_path, task = "SemEval")

evolutionary_prompts = evo.extract_lines_to_dict("INITIAL_PROMPTS/evolutionary_prompts", task = "Evo_prompts")


"""evo.alg_baseline('SemEval', small_initial_population_prompts, evolutionary_prompts,
                model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                quantize_model_4bits = True,
                n_pop = 2, # initial population size and the number of elements kepts at each iteration
                n_top = 0, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                patience = 1,
                max_iter = 3,
                temperature = 1.0, #temperature for decoding combined and mutated
                top_p = 0.8, #sampling for decoding combined and mutated
                save = True,
                eval_data = 'dev', # dev or train
                data_size = 10,
                test_eval = False ) # no. of samples where the prompts are evaluated, if =0 all are used"""

evo.alg_baseline('SemEval', initial_population_prompts, evolutionary_prompts,
                model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                quantize_model_4bits = True,
                n_pop = 5, # initial population size and the number of elements kepts at each iteration
                n_top = 0, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                patience = 10,
                max_iter = 200,
                temperature = 1.0, #temperature for decoding combined and mutated
                top_p = 0.8, #sampling for decoding combined and mutated
                save = True,
                eval_data = 'dev', # dev or train
                data_size = 0,
                test_eval = True ) # no. of samples where the prompts are evaluated, if =0 all are used"""

"""
folder_path = 'INITIAL_PROMPTS/ContractNLI_initial_population_prompts'
small_folder_path = 'INITIAL_PROMPTS/small_ContractNLI_initial_population_prompts'

initial_population_prompts = evo.extract_lines_to_dict(folder_path, task = "ContractNLI")
small_initial_population_prompts = evo.extract_lines_to_dict(small_folder_path, task = "ContractNLI")

evo.alg_baseline('ContractNLI', initial_population_prompts, evolutionary_prompts,
                model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                quantize_model_4bits = True,
                n_pop = 5, # initial population size and the number of elements kepts at each iteration
                n_top = 0, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                patience = 10,
                max_iter = 200,
                temperature = 1.0, #temperature for decoding combined and mutated
                top_p = 0.8, #sampling for decoding combined and mutated
                save = True,
                eval_data = 'dev', # dev or train
                data_size = 0,
                test_eval = True ) # no. of samples where the prompts are evaluated, if =0 all are used"""
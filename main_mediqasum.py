import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            #model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            #model_name = "microsoft/Phi-3.5-mini-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
                                                            #model_name = "unsloth/Phi-3.5-mini-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            model_name = "meta-llama/Llama-3.2-1B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True,
                                                            )

best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            #model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            #model_name = "microsoft/Phi-3.5-mini-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
                                                            #model_name = "unsloth/Phi-3.5-mini-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            model_name = "meta-llama/Llama-3.2-1B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True,
                                                            )

"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            resume_run=True,
                                                            resume_run_folder="RUNS_alg_2/MEDIQASUM/Runs_2024-08-01_17-24-23_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue"
                                                            )"""

"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = False,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            resume_run=True,
                                                            resume_run_folder="RUNS_alg_2/MEDIQASUM/Runs_2024-07-24_04-35-31_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_new_evo_promptsTrue"
                                                            )"""





"""
evo.extract_MEDIQASUM_data(folder_name='DATASETS/MEDIQASUM_data', 
                           type = 'valid', 
                           used_retrieved_file = False,
                           retrieve_similar_examples = True,
                           save_retrieved = True,
                           )
                           
evo.extract_MEDIQASUM_data(folder_name='DATASETS/MEDIQASUM_data', 
                           type = 'train', 
                           used_retrieved_file = False,
                           retrieve_similar_examples = True,
                           save_retrieved = True,
                           )

evo.extract_MEDIQASUM_data(folder_name='DATASETS/MEDIQASUM_data', 
                           type = 'clinicalnlp_taskB_test1', 
                           used_retrieved_file = False,
                           retrieve_similar_examples = True,
                           save_retrieved = True,
                           )
"""


"""
# já para a versão normal, not baseline
evo.test_eval(task="MEDIQASUM", 
                  RUN_folder_path = 'RUNS_alg_2/MEDIQASUM/Runs_2024-08-01_17-24-23_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue', 
                  model_name="microsoft/Phi-3-mini-4k-instruct",
                  task_w_one_shot = True,
                  task_w_highlight = False,
                  task_w_self_reasoning = False,
                  ) 
"""
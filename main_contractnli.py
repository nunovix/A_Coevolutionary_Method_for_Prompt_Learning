import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo

evo.test_eval('ContractNLI',
              'RUNS_alg_2/ContractNLI_woracleTrue_w2labelsTrue/Runs_2024-09-07_03-51-39_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrueuse_dq_dataTrue', # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=False,
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              task_w_oracle_spans= True, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

# DQ 614
"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 614, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_highlight = True,
                                                            task_w_oracle_spans = True,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            use_data_sorted_by_dq = True
                                                            )"""



"""evo.extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                             type = 'test',
                             use_retrieves_sentences_files = False,
                             retrieve_sentences = True,
                             save_retrieved_sentences = True,
                             task_w_2_labels = False,
                             )"""
##
"""evo.test_eval('ContractNLI',
              'RUNS_alg_2/ContractNLI_woracleFalse_w2labelsFalse/Runs_2024-07-22_10-56-01_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue', # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=False,
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              task_w_oracle_spans= False, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = False, # contract nli only
              )"""

"""
# baseline, sem hall of fame, só com mut, não guiada
best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob=0.5,
                                                            crossover_prob=0.0,
                                                            sampling_T = None,
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
"""

"""
# w oracle spans 2 labels
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
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            )

# w retrieved spans 2 labels
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
                                                            task_w_oracle_spans = False,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            )


# w retrieved spans 3 labels
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
                                                            task_w_oracle_spans = False,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=False,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            )
"""
# code to run the experiments reported in the ContractNLI dataset

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import evo_functions as evo

# MC - Oracle - 15% Random
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
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False
                                                            )

# CoEvo - Oracle - 15% Random
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
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False
                                                            )

# CoEvo - Retrieved - 15% Random
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
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False
                                                            )

# MC - Oracle - 15% DQ
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
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True
                                                            )

# CoEvo - Oracle - 15% DQ
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
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True
                                                            )
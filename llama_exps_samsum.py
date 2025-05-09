import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo
# MC - 1% Random - Llama 3B
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False,   ### True,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 1,   ### 0.25, 
                                                            use_15percent_random=True,
                                                            use_data_sorted_by_dq=False,
                                                            reverse_dq=False,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            )
"""

# CoEvo - 1% Random - Llama 3B
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False, 
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 1,  
                                                            use_15percent_random=True,
                                                            use_data_sorted_by_dq=False,
                                                            reverse_dq=False,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            )
"""

# CoEvo - 1% Random - Llama 70B
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.3-70B-Instruct",
                                                            quantize_model_4bits = True, 
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 0,
                                                            max_iter = 0,
                                                            data_size = 1,  
                                                            use_15percent_random=True,
                                                            use_data_sorted_by_dq=False,
                                                            reverse_dq=False,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            resume_run = True,
                                                            resume_run_folder = "RUNS_alg_2/SAMSum/Runs_2025-05-08_16-20-29_N25_cp0.25_mp0.25_sampT10.0_fixed_evoTrue_dq_dataFalse_reverseFalse_dev_ratioTrue_155/"
                                                            )
"""

# MC - 1% DQ - Llama 3B
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.01,
                                                            use_15percent_random = False,
                                                            use_data_sorted_by_dq = True,
                                                            reverse_dq = True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            )
"""

# CoEvo - 1% DQ - Llama 3B
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.01,
                                                            use_15percent_random = False,
                                                            use_data_sorted_by_dq = True,
                                                            reverse_dq = True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            )


# CoEvo - 1% DQ - Llama 70B
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.3-70B-Instruct",
                                                            quantize_model_4bits = True, 
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 0,
                                                            max_iter = 0,
                                                            data_size = 0.01,  
                                                            use_15percent_random = False,
                                                            use_data_sorted_by_dq = True,
                                                            reverse_dq = True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            resume_run = True,
                                                            resume_run_folder = ""   #TODO:
                                                            )
"""
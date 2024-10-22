import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import evo_functions as evo

# contract nli com bottom 707 sem keep dev ratio
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
                                                            data_size = 707, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_highlight = True,
                                                            task_w_oracle_spans = False,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            use_data_sorted_by_dq=True,
                                                            reverse_dq=True,
                                                            keep_dev_ratio=False,
                                                            )




#FALTA CORRER
"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR", 
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
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False,
                                                            )  """





"""
# já correu
best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR",
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
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True,
                                                            resume_run = True,
                                                            resume_run_folder = "RUNS_alg_2/LegalSumTOSDR_woneshotTrue/Runs_2024-10-16_06-46-14_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomFalse_15per_revdqTrue",
                                                            )"""
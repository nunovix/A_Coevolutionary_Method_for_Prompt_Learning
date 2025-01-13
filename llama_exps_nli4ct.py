import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 5,
                                                    n_top = 1,
                                                    mutation_prob = 0.25,
                                                    crossover_prob = 0.25,
                                                    sampling_T = 10.0,
                                                    patience = 10,
                                                    max_iter = 200,
                                                    data_size = 0.25, 
                                                    use_data_sorted_by_dq=False,
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = True,
                                                    task_w_highlight = True,
                                                    fixed_evo_prompts = False,
                                                    do_test_eval = True,
                                                    save = True,
                                                    )

"""
evo.test_eval(task="SemEval",
              RUN_folder_path="RUNS_alg_2/SemEval_whighFalse_wselfFalse/Runs_2024-12-17_04-24-46_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_dq_dataFalse_reverseFalse_dev_ratioTrue_474", # Run folder
              model_name = "meta-llama/Llama-3.2-3B-Instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

evo.test_eval(task="SemEval",
              RUN_folder_path="RUNS_alg_2_DQ/SemEval_whighFalse_wselfFalse/Runs_2024-12-18_16-19-23_N25_cp0.0_mp0.5_sampTNone_fixed_evoFalse_dq_dataTrue_reverseFalse_dev_ratioTrue_474", # Run folder
              model_name = "meta-llama/Llama-3.2-3B-Instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

evo.test_eval(task="SemEval",
              RUN_folder_path="RUNS_alg_2_DQ/SemEval_whighFalse_wselfFalse/Runs_2024-12-18_16-19-23_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_dq_dataTrue_reverseFalse_dev_ratioTrue_474", # Run folder
              model_name = "meta-llama/Llama-3.2-3B-Instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )


# MC - 1-shot - 25% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.25, 
                                                            use_data_sorted_by_dq=False,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            )"""
"""
# coevo 0 shot DQ
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob = 0.25,
                                                    crossover_prob = 0.25,
                                                    sampling_T = 10.0,
                                                    patience = 10,
                                                    max_iter = 200,
                                                    data_size = 0.25, 
                                                    use_data_sorted_by_dq=True,
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = False,
                                                    task_w_highlight = False,
                                                    fixed_evo_prompts = False,
                                                    do_test_eval = True,
                                                    save = True,
                                                    )

#MC 0-shot random
"""
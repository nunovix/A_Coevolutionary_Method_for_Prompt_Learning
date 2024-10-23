import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import evo_functions as evo


# self reasoning
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                                model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                                quantize_model_4bits = True,
                                                                n_pop = 25,
                                                                n_top = 5,
                                                                mutation_prob=0.25,
                                                                crossover_prob=0.25,
                                                                sampling_T = 10.0,
                                                                patience = 10,
                                                                max_iter = 200,
                                                                data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                                task_w_one_shot = False,
                                                                task_w_self_reasoning = True,
                                                                task_w_highlight = False,
                                                                fixed_evo_prompts = True,
                                                                do_test_eval = True,
                                                                new_evo_prompt_format = True,
                                                                save = True,
                                                                use_15percent_random=False,
                                                                use_15percent_revdq=True,
                                                                )





#CORRIDO
"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
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
                                                            resume_run_folder = "RUNS_alg_2/MEDIQASUM/Runs_2024-10-16_18-00-57_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomFalse_15per_revdqTrue",
                                                            )
"""


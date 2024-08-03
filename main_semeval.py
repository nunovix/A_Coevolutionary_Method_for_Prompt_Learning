import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo

# self-reasoning
"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
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
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format = True,
                                                            save = True,
                                                            use_optimized_evo_prompts = False
                                                            )"""

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
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format = True,
                                                            save = True,
                                                            use_optimized_evo_prompts = False,
                                                            resume_run=True,
                                                            resume_run_folder="RUNS_alg_2/SemEval_whighFalse_wselfTrue/Runs_2024-07-26_17-40-27_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue"
                                                            )

"""


# base
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 5,
                                                            n_top = 1,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 20, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format = True,
                                                            save = False,
                                                            )


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
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format=True,
                                                            save=True,
                                                            )

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
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format=True,
                                                            save=True,
                                                            )

"""
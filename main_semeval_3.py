import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import evo_functions as evo

#reverse, ratio
sizes = [400, 600, 1000]
for size in sizes:
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
                                                                data_size = size, # number of examples where the prompts are evaluate 0 means all
                                                                task_w_one_shot = False,
                                                                task_w_self_reasoning = False,
                                                                task_w_highlight = False,
                                                                fixed_evo_prompts = False,
                                                                do_test_eval = True,
                                                                new_evo_prompt_format = True,
                                                                save = True,
                                                                use_data_sorted_by_dq=True,
                                                                reverse_dq=True,
                                                                keep_dev_ratio=True,
                                                                )

# normal, no ratio
sizes = [400, 600, 1000]
for size in sizes:
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
                                                                data_size = size, # number of examples where the prompts are evaluate 0 means all
                                                                task_w_one_shot = False,
                                                                task_w_self_reasoning = False,
                                                                task_w_highlight = False,
                                                                fixed_evo_prompts = False,
                                                                do_test_eval = True,
                                                                new_evo_prompt_format = True,
                                                                save = True,
                                                                use_data_sorted_by_dq=True,
                                                                reverse_dq=False,
                                                                keep_dev_ratio=False,
                                                                )
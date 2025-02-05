import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 0,
                                                    mutation_prob = 0.5,
                                                    crossover_prob = 0.0,
                                                    sampling_T = None,
                                                    patience = 10,
                                                    max_iter = 200,
                                                    data_size = 0.25, 
                                                    use_data_sorted_by_dq=True,
                                                    reverse_dq=True,
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = True,
                                                    task_w_highlight = True,
                                                    fixed_evo_prompts = False,
                                                    do_test_eval = True,
                                                    save = True,
                                                    do_test_eval_initial=False
                                                    )


import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import evo_functions as evo

"""best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR",
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
                                                            reverse_dq=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            )"""

# MC - 1-shot - 25% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR",
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
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            )
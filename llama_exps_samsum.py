import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo
# MC - 25% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False,   ### True,
                                                            n_pop = 5,   ### 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 3,   ### 10,
                                                            max_iter = 5,  ### 200,
                                                            data_size = 0.01,   ### 0.25, 
                                                            use_data_sorted_by_dq=False,
                                                            reverse_dq=False,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            )

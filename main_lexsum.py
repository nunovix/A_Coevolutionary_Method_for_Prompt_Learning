import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo
best_prompt, best_score_iterations = evo.evo_alg_2(task = "LEXSUM", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 5,
                                                            n_top = 1,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = None,
                                                            patience = 2,
                                                            max_iter = 2,
                                                            data_size = 20, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = False,
                                                            task_w_one_shot = True,
                                                            )
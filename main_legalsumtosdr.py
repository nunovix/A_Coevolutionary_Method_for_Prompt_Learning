import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 5,
                                                            n_top = 1,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 5,
                                                            max_iter = 20,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            )
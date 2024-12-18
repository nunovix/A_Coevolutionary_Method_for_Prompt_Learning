import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                                #model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                                model_name = "meta-llama/Llama-3.2-1B-Instruct",
                                                                quantize_model_4bits = True,
                                                                n_pop = 5,
                                                                n_top = 1,
                                                                mutation_prob=0.25,
                                                                crossover_prob=0.25,
                                                                sampling_T = 10.0,
                                                                patience = 10,
                                                                max_iter = 200,
                                                                data_size = 0.15, # number of examples where the prompts are evaluate 0 means all
                                                                task_w_one_shot = False,
                                                                task_w_self_reasoning = False,
                                                                task_w_highlight = False,
                                                                fixed_evo_prompts = False,
                                                                do_test_eval = True,
                                                                save = False,
                                                                )
# code to run the experiments reported in the SAMSum dataset

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo


# MC - 15% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 1, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            use_15percent_random=True,  #TODO: not used? It seems that this part is done based only on "data_size"
                                                            use_15percent_revdq=False,
                                                            )

"""
# CoEvo - 1-shot - 15% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = False,
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
                                                            task_w_one_shot = False,
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False,
                                                            )

# MC - 1-shot - 15% DQ
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 0,
                                                            mutation_prob = 0.5,
                                                            crossover_prob = 0.0,
                                                            sampling_T = None,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True,
                                                            )

# CoEvo -1-shot - 15% DQ
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = False,
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
                                                            task_w_one_shot = False,
                                                            use_15percent_random=False,
                                                            use_15percent_revdq=True,
                                                            )
"""
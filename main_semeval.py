import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo
#

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob=0.75,
                                                    crossover_prob=0.5,
                                                    sampling_T = 10.0,
                                                    patience = 30,
                                                    max_iter = 200,
                                                    data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = False,
                                                    task_w_highlight = False,
                                                    fixed_evo_prompts=False,
                                                    do_test_eval = True,
                                                    )

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob=0.75,
                                                    crossover_prob=0.5,
                                                    sampling_T = 10.0,
                                                    patience = 30,
                                                    max_iter = 200,
                                                    data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = False,
                                                    task_w_highlight = True,
                                                    fixed_evo_prompts=False,
                                                    do_test_eval = True,
                                                    )



"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob=0.75,
                                                            crossover_prob=0.5,
                                                            sampling_T = 10.0,
                                                            patience = 30,
                                                            max_iter = 200,
                                                            data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = False,
                                                            do_test_eval = True,
                                                            ) #


best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob=0.75,
                                                    crossover_prob=0.5,
                                                    sampling_T = 10.0,
                                                    patience = 30,
                                                    max_iter = 200,
                                                    data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = False,
                                                    task_w_highlight = True,
                                                    do_test_eval = True,
                                                    )
"""




"""
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                   model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                   #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                   quantize_model_4bits = True,
                                                   n_pop = 5,
                                                   n_top = 1,
                                                   mutation_prob=0.5,
                                                   crossover_prob=0.5,
                                                   sampling_T = 5.0,
                                                   patience = 2,
                                                   max_iter = 2,
                                                   data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                   task_w_one_shot = False,
                                                   task_w_self_reasoning = True,
                                                   task_w_highlight = False,
                                                   do_test_eval = False,
                                                   save=True,
                                                   ) 

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                   model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                   #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                   quantize_model_4bits = True,
                                                   n_pop = 5,
                                                   n_top = 1,
                                                   mutation_prob=0.5,
                                                   crossover_prob=0.5,
                                                   sampling_T = 5.0,
                                                   patience = 3,
                                                   max_iter = 5,
                                                   data_size = 10, # number of examples where the prompts are evaluate 0 means all
                                                   task_w_one_shot = False,
                                                   task_w_self_reasoning = False,
                                                   task_w_highlight = True,
                                                   do_test_eval = False,
                                                   save=True,
                                                   ) #
"""
import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import evo_functions as evo
"""
n_pop = 10
exp_mutation_prob = [0.75,]
exp_crossover_prob = [0.25, 0.5, 0.75]
exp_samp_T = [1.0, 5.0, 10.0]

# grid search for hyperparameters
for m_p in exp_mutation_prob:
    for c_p in exp_crossover_prob:
        for s_T in exp_samp_T:

            best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                            model_name = "microsoft/Phi-3-mini-128k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = n_pop,
                                                            n_top = 1,
                                                            mutation_prob=m_p,
                                                            crossover_prob=c_p,
                                                            sampling_T = s_T,
                                                            patience = 30,
                                                            max_iter = 200,
                                                            data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = False,
                                                            do_test_eval = True,
                                                            ) #"""


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
                                                            ) #"""
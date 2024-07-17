import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo

evo.test_eval("SemEval", 
          'RUNS_alg_2/SemEval_whighTrue_wselfFalse/Runs_2024-07-15_03-42-41_N25_cp0.25_mp0.25_sampT10.0_fixed_evoTrue_new_evo_promptsTrue',
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_preds4semeval_test=False, # is turned to true if it's the semeval task
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              )

evo.test_eval("SemEval", 
          'RUNS_alg_2/SemEval_whighTrue_wselfFalse/Runs_2024-07-15_00-33-02_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue',
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_preds4semeval_test=False, # is turned to true if it's the semeval task
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              )

evo.test_eval("SemEval", 
          'RUNS_alg_2/SemEval_whighTrue_wselfFalse/Runs_2024-07-14_04-32-07_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue',
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_preds4semeval_test=False, # is turned to true if it's the semeval task
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              )


"""


# base
best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                            model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 5,
                                                            n_top = 1,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 20, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format = True,
                                                            save = False,
                                                            )


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
                                                            data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format=True,
                                                            save=True,
                                                            )

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
                                                            data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                            task_w_one_shot = False,
                                                            task_w_self_reasoning = False,
                                                            task_w_highlight = True,
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            new_evo_prompt_format=True,
                                                            save=True,
                                                            )

"""
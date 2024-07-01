import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo

best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "microsoft/Phi-3-mini-4k-instruct",
                                                    #model_name = "unsloth/Phi-3-mini-4k-instruct",
                                                    #model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob=0.75,
                                                    crossover_prob=0.5,
                                                    sampling_T = 10.0,
                                                    patience = 10,
                                                    max_iter = 200,
                                                    data_size = 0, # number of examples where the prompts are evaluate 0 means all
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = True,
                                                    task_w_highlight = False,
                                                    fixed_evo_prompts=True,
                                                    do_test_eval = True,
                                                    new_evo_prompt_format=True,
                                                    save=True,
                                                    )

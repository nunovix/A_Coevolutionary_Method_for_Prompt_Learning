import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#os.environ["TOKENIZERS_PARALLELISM"] = "false"

import evo_functions as evo


# MC - 1-shot - 15% Random
best_prompt, best_score_iterations = evo.evo_alg_2(task = "MEDIQASUM", 
                                                            model_name = "meta-llama/Llama-3.2-1B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 5,
                                                            n_top = 1,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0, 
                                                            fixed_evo_prompts = True,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            use_15percent_random=True,
                                                            use_15percent_revdq=False,
                                                            )                                                                
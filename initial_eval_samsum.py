import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import evo_functions as evo

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "SAMSum", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = False,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.025, 
                                                            use_data_sorted_by_dq = True,
                                                            reverse_dq = True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = False,
                                                            do_test_eval_initial = True,
                                                            )
except:
    print("Error in SAMSum")
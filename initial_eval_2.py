import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import evo_functions as evo

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR",
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.25,
                                                            use_data_sorted_by_dq=False,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            do_test_eval_initial = True,
                                                            )
except:
    print("Error in LegalSumTOSDR")

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "LegalSumTOSDR",
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob = 0.25,
                                                            crossover_prob = 0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.25,
                                                            use_data_sorted_by_dq=True,
                                                            reverse_dq=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            task_w_one_shot = True,
                                                            do_test_eval_initial = True,
                                                            )
except:
    print("Error in LegalSumTOSDR")



"""try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                        model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                        quantize_model_4bits = True,
                                                        n_pop = 25,
                                                        n_top = 5,
                                                        mutation_prob = 0.25,
                                                        crossover_prob = 0.25,
                                                        sampling_T = 10.0,
                                                        patience = 10,
                                                        max_iter = 200,
                                                        data_size = 0.25, 
                                                        use_data_sorted_by_dq=False,
                                                        task_w_one_shot = False,
                                                        task_w_self_reasoning = False,
                                                        task_w_highlight = False,
                                                        fixed_evo_prompts = False,
                                                        do_test_eval = True,
                                                        save = True,
                                                        do_test_eval_initial=True
                                                        )
except:
    pass

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "SemEval", 
                                                    model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                    quantize_model_4bits = True,
                                                    n_pop = 25,
                                                    n_top = 5,
                                                    mutation_prob = 0.25,
                                                    crossover_prob = 0.25,
                                                    sampling_T = 10.0,
                                                    patience = 10,
                                                    max_iter = 200,
                                                    data_size = 0.25, 
                                                    use_data_sorted_by_dq=True,
                                                    reverse_dq=True,
                                                    task_w_one_shot = False,
                                                    task_w_self_reasoning = False,
                                                    task_w_highlight = False,
                                                    fixed_evo_prompts = False,
                                                    do_test_eval = True,
                                                    save = True,
                                                    do_test_eval_initial=True
                                                    )
except:
    pass

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.25, # number of examples where the prompts are evaluate 0 means all
                                                            use_data_sorted_by_dq=False,
                                                            task_w_highlight = True,
                                                            task_w_oracle_spans = True,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            do_test_eval_initial=True,
                                                            )
except:
    pass

try:
    best_prompt, best_score_iterations = evo.evo_alg_2(task = "ContractNLI", 
                                                            model_name = "meta-llama/Llama-3.2-3B-Instruct",
                                                            quantize_model_4bits = True,
                                                            n_pop = 25,
                                                            n_top = 5,
                                                            mutation_prob=0.25,
                                                            crossover_prob=0.25,
                                                            sampling_T = 10.0,
                                                            patience = 10,
                                                            max_iter = 200,
                                                            data_size = 0.25, # number of examples where the prompts are evaluate 0 means all
                                                            use_data_sorted_by_dq=True,
                                                            reverse_dq=True,
                                                            task_w_highlight = True,
                                                            task_w_oracle_spans = True,
                                                            task_w_full_contract = True,
                                                            task_w_2_labels=True,
                                                            fixed_evo_prompts = False,
                                                            do_test_eval = True,
                                                            save = True,
                                                            do_test_eval_initial=True,
                                                            )
except:
    pass"""


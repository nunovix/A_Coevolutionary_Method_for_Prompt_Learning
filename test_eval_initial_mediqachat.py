import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import evo_functions as evo

path = 'For_Initial_pop_test_eval/dq_15percent_exps/MEDIQA/CoEvo_DQ_Runs_2024-10-16_18-00-57_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomFalse_15per_revdqTrue'
evo.test_eval('MEDIQASUM',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

path = 'For_Initial_pop_test_eval/dq_15percent_exps/MEDIQA/MC_DQ_Runs_2024-10-17_01-26-57_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_15per_randomFalse_15per_revdqTrue'
evo.test_eval('MEDIQASUM',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

path = 'For_Initial_pop_test_eval/random_data_Experiments/MEDIQA/base_Runs_2024-08-01_17-24-23_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue'
evo.test_eval('MEDIQASUM',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

path = 'For_Initial_pop_test_eval/random_data_Experiments/MEDIQA/baseline_Runs_2024-07-24_04-35-31_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_new_evo_promptsTrue'
evo.test_eval('MEDIQASUM',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )
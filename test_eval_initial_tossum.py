import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import evo_functions as evo

path = 'For_Initial_pop_test_eval/dq_15percent_exps/ToSSum/CoEvo_DQ_Runs_2024-10-16_06-46-14_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomFalse_15per_revdqTrue'
evo.test_eval('LegalSumTOSDR',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

path = 'For_Initial_pop_test_eval/dq_15percent_exps/ToSSum/CoEvo_random_Runs_2024-10-20_03-04-17_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomTrue_15per_revdqFalse'
evo.test_eval('LegalSumTOSDR',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )


path = 'For_Initial_pop_test_eval/dq_15percent_exps/ToSSum/MC_DQ_Runs_2024-10-21_00-21-04_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_15per_randomFalse_15per_revdqTrue'
evo.test_eval('LegalSumTOSDR',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

path = 'For_Initial_pop_test_eval/dq_15percent_exps/ToSSum/MC_random_Runs_2024-10-23_15-45-42_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_15per_randomTrue_15per_revdqFalse'
evo.test_eval('LegalSumTOSDR',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_one_shot = True,
              )

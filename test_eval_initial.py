import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import evo_functions as evo

"""
path = 'For_Initial_pop_test_eval/dq_15percent_exps/ContractNLI/CoEvo_oracle_DQ_Runs_2024-09-21_22-13-52_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalsedq_dataTrue_reverseTrue_dev_ratioFalse_614_clusterFalse'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= True, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

path = 'For_Initial_pop_test_eval/dq_15percent_exps/ContractNLI/CoEvo_retrieved_DQ_Runs_2024-10-20_03-03-08_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalsedq_dataTrue_reverseTrue_dev_ratioFalse_707_clusterFalse'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= False, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

path = 'For_Initial_pop_test_eval/dq_15percent_exps/ContractNLI/MC_oracle_DQ_Runs_2024-10-11_04-05-14_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_15per_randomFalse_15per_revdqTrue'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= True, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

"""
#############

path = 'For_Initial_pop_test_eval/random_data_Experiments/ContractNLI/baseline_Runs_2024-07-17_02-33-36_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_new_evo_promptsTrue'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= True, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

path = 'For_Initial_pop_test_eval/random_data_Experiments/ContractNLI/retrieved_Runs_2024-07-21_12-18-30_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= False, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )

path = 'For_Initial_pop_test_eval/random_data_Experiments/ContractNLI/oracle_Runs_2024-09-18_00-36-00_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalsedq_dataFalse_reverseFalse_dev_ratioFalse_614'
evo.test_eval('ContractNLI',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_highlight = True,
              task_w_oracle_spans= True, # contract nli only
              task_w_full_contract = True, # contract nli only
              task_w_2_labels = True, # contract nli only
              )
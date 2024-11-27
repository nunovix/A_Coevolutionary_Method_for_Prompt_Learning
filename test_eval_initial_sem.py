import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import evo_functions as evo

"""
path = 'For_Initial_pop_test_eval/dq_15percent_exps/SemEval/MC_base_DQ_Runs_2024-10-11_04-03-36_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_15per_randomFalse_15per_revdqTrue'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

#################################################
path = 'For_Initial_pop_test_eval/dq_15percent_exps/SemEval/CoEvo_base_DQ_Runs_2024-10-10_13-40-00_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_dq_dataTrue_reverseTrue_dev_ratioTrue_300_clusterFalse_from_None'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

#################################################
path = 'For_Initial_pop_test_eval/dq_15percent_exps/SemEval/CoEvo_high_DQ_Runs_2024-10-10_21-26-20_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_dq_dataTrue_reverseTrue_dev_ratioTrue_300_clusterFalse_from_None'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              )

#################################################
path = 'For_Initial_pop_test_eval/random_data_Experiments/SemEval/Baseline_Runs_2024-07-11_13-57-45_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_new_evo_promptsTrue'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

#################################################
path = 'For_Initial_pop_test_eval/random_data_Experiments/SemEval/Base_Runs_2024-09-24_17-20-15_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

#################################################
path = 'For_Initial_pop_test_eval/random_data_Experiments/SemEval/after_hyper_base_Runs_2024-07-27_21-42-09_N25_cp0.25_mp0.25_sampT10.0_fixed_evoTrue_new_evo_promptsTrue'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              )

#################################################
path = 'For_Initial_pop_test_eval/random_data_Experiments/SemEval/highlights_Runs_2024-07-14_04-32-07_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = True,
              )
"""
#################################################
path = 'For_Initial_pop_test_eval/random_data_Experiments/SemEval/self_Runs_2024-07-26_17-40-27_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue'
evo.test_eval('SemEval',
              path, # Run folder
              model_name = "microsoft/Phi-3-mini-4k-instruct",
              quantize_model_4bits = True,
              save_test_predictions=True,
              task_w_self_reasoning = True,
              task_w_one_shot = False,
              task_w_highlight = False,
              )
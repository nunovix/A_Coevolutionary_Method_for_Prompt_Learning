import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from evo_functions import test_eval, create_plots_from_RUNS_folder

#test_eval(task='ContractNLI', RUN_folder_path = 'RUNS_ContractNLI/Runs_2024-04-16_16-13-43', model_name = "microsoft/Phi-3-mini-128k-instruct",)

#test_eval(task='SemEval_self', RUN_folder_path = "RUNS/SemEval_self/Runs_2024-05-03_12-21-44", model_name = "microsoft/Phi-3-mini-128k-instruct",)

#test_eval(task='SemEval', RUN_folder_path = "RUNS/SemEval/Runs_2024-05-31_04-32-35", model_name = "microsoft/Phi-3-mini-128k-instruct", )

#test_eval(task='MEDIQASUM', RUN_folder_path = "RUNS/MEDIQASUM/Runs_2024-05-19_00-44-39", model_name="microsoft/Phi-3-mini-4k-instruct")



folder = 'RUNS_alg_2/SemEval_whighFalse_wselfFalse/Runs_2024-09-08_08-07-36_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue_use_dq_dataTrue_600'
#create_plots_from_RUNS_folder(folder)"""

test_eval(task='SemEval', RUN_folder_path = folder, model_name = "microsoft/Phi-3-mini-4k-instruct")

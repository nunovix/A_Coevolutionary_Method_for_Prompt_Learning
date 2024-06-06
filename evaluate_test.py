import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from evo_functions import test_eval
from semeval_evaluation import main

#test_eval(task='ContractNLI', RUN_folder_path = 'RUNS_ContractNLI/Runs_2024-04-16_16-13-43', model_name = "microsoft/Phi-3-mini-128k-instruct",)

#test_eval(task='SemEval_self', RUN_folder_path = "RUNS/SemEval_self/Runs_2024-05-03_12-21-44", model_name = "microsoft/Phi-3-mini-128k-instruct",)

test_eval(task='SemEval', RUN_folder_path = "RUNS/SemEval/Runs_2024-05-31_04-32-35", model_name = "microsoft/Phi-3-mini-128k-instruct", )

#test_eval(task='MEDIQASUM', RUN_folder_path = "RUNS/MEDIQASUM/Runs_2024-05-19_00-44-39", model_name="microsoft/Phi-3-mini-4k-instruct")

"""main(pred_filename='RUNS/SemEval/Runs_2024-05-15_00-49-56/Iteration_best/test_predictions.json',
     gold_filename='DATASETS/SemEval_data/gold_test.json',
     output_dir='RUNS/SemEval/Runs_2024-05-15_00-49-56/Iteration_best'
     )"""


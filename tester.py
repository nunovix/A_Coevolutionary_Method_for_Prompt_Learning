import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from evo_functions import test_eval

test_eval("SemEval",
          RUN_folder_path = "RUNS_alg_2/SemEval_whighFalse_wselfFalse/Runs_2024-12-17_02-34-12_N5_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_15per_randomFalse_15per_revdqFalse",
          model_name = "meta-llama/Llama-3.3-70B-Instruct",
          quantize_model_4bits = True,
          save_test_predictions=True,
          task_w_self_reasoning = False,
          task_w_highlight = False,
          )
import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from data_quality_functions import data_quality_assessment_and_save

data_quality_assessment_and_save(task = 'SemEval')
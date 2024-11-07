# code to run the DQ assessment used to filter the data in the reported experiments

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from data_quality_functions import data_quality_assessment_and_save

# DQ assessment for NLI4CT dataset
data_quality_assessment_and_save(task = 'SemEval', save=False)

# DQ assessment for ContractNLI dataset
data_quality_assessment_and_save(task = 'ContractNLI', save=True, phi_model='128k')

# DQ assessment for MEDIQA-CHAT dataset
data_quality_assessment_and_save(task = 'MEDIQASUM', save=True)

# DQ assessment for ToS-Sum dataset
data_quality_assessment_and_save(task = 'LegalSumTOSDR', save=True)

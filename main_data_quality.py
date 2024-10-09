import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_quality_functions import data_quality_assessment_and_save

#data_quality_assessment_and_save(task = 'SemEval', save=False)

#data_quality_assessment_and_save(task = 'ContractNLI', save=True, phi_model='128k')

#data_quality_assessment_and_save(task = 'MEDIQASUM', save=True)

data_quality_assessment_and_save(task = 'LegalSumTOSDR', save=True)

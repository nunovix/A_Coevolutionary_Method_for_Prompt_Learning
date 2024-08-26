import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from evo_functions import extract_SemEval_data, extract_ContractNLI_data, extract_MEDIQASUM_data
"""
extract_SemEval_data(folder = 'DATASETS/SemEval_data', 
                         type = 'train', 
                         extract_examples = False,
                         use_retrieves_sentences_files = False,
                         retrieve_sentences = True,
                         save_retrieved_sentences = True)"""


"""extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                             type = 'train',
                             use_retrieves_sentences_files = False,
                             retrieve_sentences = True,
                             save_retrieved_sentences = True,
                             task_w_2_labels = False, # for the experience with the oracle spans the results in the task's paper are only reported with 2 classes, excluding the NotMentioned one. that's why this flag is needed
                             )"""

extract_MEDIQASUM_data(type = 'train')
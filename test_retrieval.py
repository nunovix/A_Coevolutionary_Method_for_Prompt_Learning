import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from evo_functions import extract_ContractNLI_data

data = extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                                type = 'dev',
                                use_retrieves_sentences_files = False,
                                retrieve_sentences = True,
                                save_retrieved_sentences = True,
                                task_w_2_labels=False,
                                )



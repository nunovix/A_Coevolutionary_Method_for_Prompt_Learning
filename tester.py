import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from evo_functions import extract_SemEval_data, extract_ContractNLI_data, extract_MEDIQASUM_data, extract_LEXSUM_data


a = extract_SemEval_data(use_data_sorted_by_dq=True)
b = extract_ContractNLI_data(use_data_sorted_by_dq=True)
c = extract_MEDIQASUM_data(use_data_sorted_by_dq=True)


print(a[0].keys())
print(b[0].keys())
print(c[0].keys())


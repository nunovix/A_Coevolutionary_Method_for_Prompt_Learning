import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


from evo_functions import extract_ContractNLI_data

data = extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                                type = 'dev',
                                use_retrieves_sentences_files = False,
                                retrieve_sentences = False,
                                save_retrieved_sentences = False,
                                task_w_oracle_spans = False,
                                )











"""
from evo_functions import extract_ContractNLI_data

data = extract_ContractNLI_data(folder = 'DATASETS/ContractNLI_data', 
                         type = 'dev', 
                         #extract_examples = False,
                         use_retrieves_sentences_files = False,
                         retrieve_sentences = True,
                         save_retrieved_sentences = True
                         )



"""



















"""
from evo_functions import extract_MEDIQASUM_data

data = extract_MEDIQASUM_data(folder_name='DATASETS/MEDIQASUM_data',  
                              used_retrieved_file = True,
                              retrieve_similar_examples = False,
                              save_retrieved = False)
"""




"""
import json

# Load from a JSON file
with open('DATASETS/small_MEDIQASUM_data/valid_w_retrieved.json', 'r') as file:
    loaded_data = json.load(file)

# Verify the loaded data
print(loaded_data)
"""




















""""
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np

def embed_texts(model_name, texts):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(texts)
    return embeddings

def compute_similarity_matrix(embeddings):
    num_texts = len(embeddings)
    similarity_matrix = np.zeros((num_texts, num_texts))
    
    for i in range(num_texts):
        for j in range(num_texts):
            if i != j:
                similarity_matrix[i, j] = cos_sim(embeddings[i], embeddings[j]).item()
            else:
                similarity_matrix[i, j] = -np.inf  # Exclude self-similarity
    
    return similarity_matrix

def find_closest_embedding(similarity_matrix, index):
    closest_index = np.argmax(similarity_matrix[index])
    similarity_score = similarity_matrix[index, closest_index]
    return closest_index, similarity_score

# Example usage
texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "Beijing",
    "sorting algorithms"
]

model_name = 'Alibaba-NLP/gte-large-en-v1.5'

# Embed all texts
embeddings = embed_texts(model_name, texts)

# Compute similarity matrix
similarity_matrix = compute_similarity_matrix(embeddings)

# Cycle through each text and find the closest embedding
for index_to_check in range(len(texts)):
    closest_index, similarity_score = find_closest_embedding(similarity_matrix, index_to_check)
    print(f"The closest embedding to '{texts[index_to_check]}' is '{texts[closest_index]}' with a similarity score of {similarity_score:.2f}")
"""
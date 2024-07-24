import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datasets import load_dataset
import json
import numpy as np
from sentence_transformers.util import cos_sim
from evo_functions import embed_texts
from tqdm import tqdm

def extract_LEXSUM_data(folder_name='DATASETS/LEXSUM_data', 
                        type = 'validation', # possible types validation and test
                        used_retrieved_file = True,):
    
    file_path = os.path.join(folder_name, f"{type}_w_retrieved.json")
    if used_retrieved_file == True and os.path.exists(file_path):
        # Load from a JSON file
        with open(file_path, 'r') as file:
            data_list = json.load(file)
        print(f"Used data with already retrieved examples from {file_path}")
        return data_list
        
    dataset = load_dataset("allenai/multi_lexsum", name="v20220616")

    # Define the column to check for None values and the columns to keep
    column_to_check = 'summary/short'
    columns_to_keep = ['id', 'sources', 'summary/short']

    dataset_dict = {}
    for split in dataset:
        # Filter and select columns
        dataset_dict[split] = [
            {key: row[key] for key in columns_to_keep}
            for row in dataset[split] if row[column_to_check] is not None
        ]
        # join strings of contract
        for i in range(len(dataset_dict[split])):
            dataset_dict[split][i]['sources'] = " ".join(dataset_dict[split][i]['sources'])

    train_sources_list = []
    for example in dataset_dict['train']:
        train_sources_list.append(example["sources"])
    print(f"len(train_sources_list)-->{len(train_sources_list)}")

    print(f"Embedding training data...")
    train_embeddings = embed_texts(train_sources_list)

    for i in tqdm(range(len(dataset_dict['validation'])), desc="validation"):
        validation_embedding = embed_texts([dataset_dict['validation'][i]['sources']])
        similarities = cos_sim(validation_embedding, train_embeddings)
        closest_index = np.argmax(similarities)
        dataset_dict['validation'][i]['retrieved_sources'] = dataset_dict['train'][closest_index]['sources']
        dataset_dict['validation'][i]['retrieved_summary/short'] = dataset_dict['train'][closest_index]['summary/short']

    # Save to a JSON file
    save_path = os.path.join(folder_name, f"validation_w_retrieved.json")
    with open(save_path, 'w') as file:
        json.dump(dataset_dict['validation'], file)
    print(f"Examples with retreival svaed to {save_path}")

    for i in tqdm(range(len(dataset_dict['test'])), desc='test'):
        test_embedding = embed_texts([dataset_dict['test'][i]['sources']])
        similarities = cos_sim(test_embedding, train_embeddings)
        closest_index = np.argmax(similarities)
        dataset_dict['test'][i]['retrieved_sources'] = dataset_dict['train'][closest_index]['sources']
        dataset_dict['test'][i]['retrieved_summary/short'] = dataset_dict['train'][closest_index]['summary/short']


    # Save to a JSON file
    save_path = os.path.join(folder_name, f"test_w_retrieved.json")
    with open(save_path, 'w') as file:
        json.dump(dataset_dict['test'], file)
    print(f"Examples with retreival svaed to {save_path}")


extract_LEXSUM_data(used_retrieved_file=False)
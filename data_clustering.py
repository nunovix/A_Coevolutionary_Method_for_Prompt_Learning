# functions to perform k means clustering on the several tasks. starting from a data file already with the data quality part

import sys
import json
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# tf-idf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# hdbscan clustering method
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

from evo_functions import extract_SemEval_data, embed_texts


# tf-idf preprocessing
# lowercasing, punctuation, numerical values and stop word removal and lemmatization
def preprocess(text):
    # nltk downloads
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # remove numerical values
    text = re.sub(r'\d+', '', text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # Join tokens back into a single string
    return ' '.join(tokens)

# perform tf idf, and return np array with results
def apply_tf_idf(texts):
    # pre-process
    preprocessed_corpus = [preprocess(doc) for doc in texts]

    # tf-idf
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_corpus)

    vectorized_texts = X.toarray()
    return vectorized_texts

# convert data instance to string to be embeded
def text_semeval(example: dict) -> str:

    prim = ''.join(example['primary_evidence'])
    if 'secondary_evidence' in example.keys():
        sec = ''.join(example['secondary_evidence'])
        text = prim + sec + example['statement']
    else:
        text = prim + example['statement']
    
    return text

# function that performs k means clustering on data from one of the four tasks
# saves data with cluster no. to DATASETS/DATA_QUALITY_w_CLUSTERS

def clustering(task: str,
               clustering_method: str = 'kmeans',
               representation: str = 'embeddings',
               n_clusters: int = 100,
               selection_method: str = 'max_dq', # options 'max_dq', 'min_dq', 'hdbscan_sampling'
               save: bool = False):
    
    if task == 'SemEval':
        data = extract_SemEval_data(use_data_sorted_by_dq=True)
        text_constructor = text_semeval
        no_examples = 200
    elif task == 'ContractNLI':
        pass
    elif task == 'SemEval':
        pass
    elif task == 'SemEval':
        pass
    else:
        sys.exit("Invalid task name")

    texts=[]
    for example in data:
        text = text_constructor(example)
        #print(text)
        texts.append(text)

    if representation == 'embeddings':
       # embed texts
        vectorized_texts = embed_texts(texts)
    elif representation == 'tf_idf':
        vectorized_texts = apply_tf_idf(texts)
    else:
        sys.exit('Invalid representation selected')
    
    #print(f"vectorized_texts-->{vectorized_texts}")

    if clustering_method == 'kmeans':

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(vectorized_texts)

        cluster_labels = kmeans.labels_
    
    elif clustering_method == 'hdbscan':
        # Compute cosine distance matrix
        cosine_dist_matrix = pairwise_distances(vectorized_texts, metric='cosine')

        # Fit HDBSCAN using precomputed distances
        hdb = HDBSCAN(metric='precomputed')
        hdb.fit(cosine_dist_matrix)
        
        cluster_labels = hdb.labels_
        cluster_probbabilities = hdb.probabilities_
        #print(f"cluster_labels-->{cluster_labels}")
        #print(f"cluster_probbabilities-->{cluster_probbabilities.tolist()}")

    elif clustering_method == 'gmm':
        pass
    else:
        sys.exit(' Invalid clustering method selected')

    #print("Cluster labels for each embedding:", cluster_labels)
    #print(Counter(cluster_labels))

    # assign cluster no. to the data
    for example, cluster, cluster_prob in zip(data, cluster_labels, cluster_probbabilities):
        example['cluster'] = int(cluster)
        example['hdbscan_prob'] = cluster_prob

    # group by clusters
    cluster_info = group_clusters(data)
    #print(cluster_info)

    # select from the clusters grouping
    select_data = round_robin_cluster_selection(cluster_info, data, 200, selection_method = selection_method)
    #print(select_data)


    if save == True:
        file_name = f'DATASETS/DATA_QUALITY_w_CLUSTERS/{task}/' + f"{representation}_{clustering_method}_{selection_method}.json"
        with open(file_name, 'w') as json_file:
            json.dump(select_data, json_file, indent=4)
        
        print(f"Data with data quality assessment and cluster info saved to {file_name}!")
    
    return data, cluster_info

# group up cluster info
# returns dict of dict. each key is the cluster number. each dict contains a list of the indexes of datapoints, 
# hdbscan assignment probs and dq score
def group_clusters(data):
    cluster_info = {}
    for i, entry in enumerate(data):
        if entry['cluster'] >= -4: # some points get assigned -1 if consideres noise for example
            if entry['cluster'] not in cluster_info: 
                cluster_info[entry['cluster']] = {'datapoint': [i], 'hdbscan_prob': [entry['hdbscan_prob']], 'dq_score': [entry['score']]}
                print(cluster_info)
            else:
                cluster_info[entry['cluster']]['datapoint'].append(i)
                cluster_info[entry['cluster']]['hdbscan_prob'].append(entry['hdbscan_prob'])
                cluster_info[entry['cluster']]['dq_score'].append(entry['score'])

    return cluster_info

def softmax(lst):
    exp_values = [math.exp(x) for x in lst]
    total = sum(exp_values)
    return [x / total for x in exp_values]


# iterates through the clusters. add datapoints based on chosen selection criterion
# selection by max dq value per cluster
# selection by min dq value in cluster
# selection by smpling from the assignment porbabilities
def round_robin_cluster_selection(cluster_info, data, no_points, selection_method):
    random.seed(42)
    select_data = []
    while len(select_data) < no_points:
        for cluster_number in cluster_info:
            if len(select_data) < no_points and len(cluster_info[cluster_number]['dq_score']) > 0:
                if selection_method == 'max_dq':
                    #print(f"{len(cluster_info[cluster_number]['dq_score'])}\n")
                    index_to_add = cluster_info[cluster_number]['dq_score'].index(max(cluster_info[cluster_number]['dq_score']))
                elif selection_method == 'min_dq':
                    index_to_add = cluster_info[cluster_number]['dq_score'].index(min(cluster_info[cluster_number]['dq_score']))
                elif selection_method == 'hdbscan_sampling':
                    softmax_weights = softmax(cluster_info[cluster_number]['hdbscan_prob'])
                    index_to_add = random.choices(range(len(softmax_weights)), weights=softmax_weights, k=1)[0]
                else:
                    sys.exit('Invalid selection method')

                # add selected datapoint
                select_data.append(data[cluster_info[cluster_number]['datapoint'][index_to_add]])

                # remove values from the cluster info dict
                del cluster_info[cluster_number]['datapoint'][index_to_add]
                #print(f"\n\ncluster_info[cluster_number]['hdbscan_prob']")
                del cluster_info[cluster_number]['hdbscan_prob'][index_to_add]
                del cluster_info[cluster_number]['dq_score'][index_to_add]
        
            if not any(cluster_info[clust]['dq_score'] for clust in cluster_info):
                break

    return select_data
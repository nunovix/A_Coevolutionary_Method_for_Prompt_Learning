# functions to perform k means clustering on the several tasks. starting from a data file already with the data quality part

import sys
import json
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

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
    return

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
               save: bool = False):
    
    if task == 'SemEval':
        data = extract_SemEval_data(use_data_sorted_by_dq=True)
        text_constructor = text_semeval
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

    if clustering_method == 'kmeans':

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(vectorized_texts)

        cluster_labels = kmeans.labels_
    
    elif clustering_method == 'hdbscan':
        # Compute cosine distance matrix
        cosine_dist_matrix = pairwise_distances(X, metric='cosine')

        # Fit HDBSCAN using precomputed distances
        hdb = HDBSCAN(metric='precomputed')
        hdb.fit(cosine_dist_matrix)
        
        cluster_labels = hdb.labels_

    elif clustering_method == 'gmm':
        # Range of possible clusters to try
        cluster_range = range(2, 100)  # Trying between 2 and 10 clusters
        bics = []

        # Fit GMM for each number of clusters and compute BIC
        for n_clusters in cluster_range:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(vectorized_texts)
            bics.append(gmm.bic(vectorized_texts))

        # BIC plots
        plt.plot(cluster_range, bics, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC')
        plt.title('BIC vs Number of Clusters')

        # Save
        plt.savefig("bic_plot.png")
        plt.show()

        # select optimal
        optimal_clusters = cluster_range[np.argmin(bics)]
        print(f"Optimal number of clusters: {optimal_clusters}")

        # fit 
        final_gmm = GaussianMixture(n_components=optimal_clusters, random_state=42)
        final_gmm.fit(vectorized_texts)

        cluster_labels = final_gmm.predict(vectorized_texts)

        # save
        np.save("cluster_labels.npy", cluster_labels)
    
    else:
        sys.exit(' Invalid clustering method')

    
    print("Cluster labels for each embedding:", cluster_labels)
    print(Counter(cluster_labels))

    # assign cluster no. to the data
    for example, cluster in zip(data, cluster_labels):
        example['cluster'] = int(cluster)
    
    if save == True:
        file_name = f'DATASETS/DATA_QUALITY_w_CLUSTERS/{task}/' + f"{clustering_method}.json"
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f"Data with data quality assessment and cluster info saved to {file_name}!")
    
    return None
# functions to perform k means clustering on the several tasks. starting from a data file already with the data quality part

import sys
import json
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np

from evo_functions import extract_SemEval_data, embed_texts

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
               clustering_method: str,
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
        sys.exit("Incorrect task")

    texts=[]
    for example in data:
        text = text_constructor(example)
        #print(text)
        texts.append(text)

    # embed texts
    embeded_texts = embed_texts(texts)

    if clustering_method == 'kmeans':
        n_clusters = 100  

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeded_texts)

        cluster_labels = kmeans.labels_

    elif clustering_method == 'gmm':
        # Range of possible clusters to try
        cluster_range = range(2, 100)  # Trying between 2 and 10 clusters
        bics = []

        # Fit GMM for each number of clusters and compute BIC
        for n_clusters in cluster_range:
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm.fit(embeded_texts)
            bics.append(gmm.bic(embeded_texts))

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
        final_gmm.fit(embeded_texts)

        cluster_labels = final_gmm.predict(embeded_texts)

        # save
        np.save("cluster_labels.npy", cluster_labels)

    
    print("Cluster labels for each embedding:", cluster_labels)
    print(Counter(cluster_labels))

    # assign cluster no. to the data
    for example, cluster in zip(data, cluster_labels):
        example['cluster'] = int(cluster)
    
    if save == True:
        file_name = 'DATASETS/DATA_QUALITY_w_CLUSTERS/' + f"{task}.json"
        with open(file_name, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        
        print(f"Data with data quality assessment and cluster info saved to {file_name}!")
    
    return None
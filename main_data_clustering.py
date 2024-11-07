# Experiments with clustering methods HDBSCAN and kmeans clustering, using data represented 
# via embeddings and TF-IDF
# Results not reported

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from data_clustering import clustering

clustering(task='SemEval',
           representation='embeddings',
           clustering_method='hdbscan',
           selection_method='max_dq',
           save=True)

clustering(task='SemEval',
           representation='embeddings',
           clustering_method='hdbscan',
           selection_method='min_dq',
           save=True)

clustering(task='SemEval',
           representation='embeddings',
           clustering_method='hdbscan',
           selection_method='hdbscan_sampling',
           save=True)

clustering(task='SemEval',
           representation='tf_idf',
           clustering_method='hdbscan',
           selection_method='max_dq',
           save=True)

clustering(task='SemEval',
           representation='tf_idf',
           clustering_method='hdbscan',
           selection_method='min_dq',
           save=True)

clustering(task='SemEval',
           representation='tf_idf',
           clustering_method='hdbscan',
           selection_method='hdbscan_sampling',
           save=True)

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from data_clustering import clustering

clustering(task='SemEval',
           clustering_method='gmm',
           save=False)
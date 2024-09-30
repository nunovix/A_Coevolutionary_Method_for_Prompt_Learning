import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from data_clustering import clustering

clustering(task='SemEval',n_clusters=2,
           save=True)
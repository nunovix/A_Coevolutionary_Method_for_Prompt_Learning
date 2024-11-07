# A Coevolutionary Method for Prompt Learning

## Abstract

## Content

- [`README.md`](README.md): Project's README file.

- [`DATASETS/`](DATASETS): Contains datasets used in the project.
- [`INITIAL_PROMPTS/`](INITIAL_PROMPTS): Contains initial population of prompts to start the evolutionary algorithm for the different tasks. The folder contains a README_promptstructures.md file explaining the prompt structure used for each task, the prompts themselves are in the folders named accordingly to the task
- [`REPORTED_EXPERIMENTS/`](REPORTED_EXPERIMENTS): Directory for reported experiment data.

- [`evo_functions.py`](evo_functions.py): Functions for evolutionary operations in the project.

- [`experiments_contractnli.py`](experiments_contractnli.py): Script to run the reported experiments in the ContractNLI dataset.
- [`experiments_mediqachat.py`](experiments_mediqachat.py): Script to run the reported experiments in the MEDIQA-CHAT dataset.
- [`experiments_nli4ct.py`](experiments_nli4ct.py): Script to run the reported experiments in the NLI4CT dataset.
- [`experiments_tossum.py`](experiments_tossum.py): Script to run the reported experiments in the ToS-Sum dataset.

- [`data_quality_functions.py`](data_quality_functions.py): Functions to assess Data Quality (DQ).
- [`main_data_quality.py`](main_data_quality.py): Main script for DQ evaluation in all 4 datasets

- [`buget_datasets_stats.ipynb`](buget_datasets_stats.ipynb): Jupyter notebook for dataset statistics. Also used to create the 15% data splits, using data selected at random and by DQ score.

- [`coevo_env.yml`](coevo_env.yml): Virtual Environment used to run the project

- [`data_clustering.py`](data_clustering.py): Functions to cluster data using HDB and kmeans method in TF-IDF and embedding representation
- [`main_data_clustering.py`](main_data_clustering.py): Main script for data clustering.



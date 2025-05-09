import os
import random
import json
from evo_functions import extract_SAMSum_data

# percent_data = 0.25
# percent_data = 0.05
percent_data = 0.01

# function to save data in the format 
def save_data2file(data, folder, file_name):
    file_name += '.json'
    save_path = os.path.join(folder, file_name)
    with open(save_path, 'w') as file:
        json.dump(data, file)
    print(f"Data saved to {save_path}")


val_samsum_data = extract_SAMSum_data(type="valid")
train_samsum_data = extract_SAMSum_data(type='train')
# test_samsum_data = extract_SAMSum_data(type='test')

full_samsum_data = train_samsum_data + val_samsum_data

print("len val data = {}".format(len(val_samsum_data)))
print("len train data = {}".format(len(train_samsum_data)))
print("len full data = {}".format(len(full_samsum_data)))
print("")

print("dataset keys:")
print(full_samsum_data[0].keys())
print("")


n_samples = int(percent_data * len(full_samsum_data))

print("Using {}% of the full_samsum_data --> n_samples = {}\n".format(
    percent_data,
    n_samples
))

# Set the seed for reproducibility
random.seed(33)

print("getting random sample ...")

samsum_n_percent_random = random.sample(full_samsum_data, n_samples)

print("len samsum_n_percent_random data = {}".format(len(samsum_n_percent_random)))
print("")

save_data2file(
    samsum_n_percent_random, 
    folder="DATASETS/{}percent_random".format(str(int(percent_data * 100))), 
    file_name='samsum'
)


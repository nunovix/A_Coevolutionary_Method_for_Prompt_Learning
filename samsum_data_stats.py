import os
from evo_functions import extract_SAMSum_data
import numpy as np
from evo_functions import load_model

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

train_samsum_data = extract_SAMSum_data(type='train')
val_samsum_data = extract_SAMSum_data(type="valid")
test_samsum_data = extract_SAMSum_data(type='test')

print("len train data = {}".format(len(train_samsum_data)))
print("len val data = {}".format(len(val_samsum_data)))
print("len test data = {}".format(len(test_samsum_data)))
print("")

print("dataset keys:")
print(val_samsum_data[0].keys())
print("")

model, tokenizer = load_model(checkpoint="meta-llama/Llama-3.2-3B-Instruct", quantized=False)

def get_n_tokens_for_data(data, tokenizer, key_name):

    assert isinstance(key_name, str)

    n_tokens_list = []

    for elem in data:
        tokenized_content = tokenizer(elem[key_name], return_tensors="pt", return_attention_mask=False)

        n_tokens_list.append(len(tokenized_content['input_ids'][0]))
    
    return n_tokens_list

def get_n_tokens_for_prompt_example():
    system_prompt = "Assume the role of an automated system for the processing of domain-specific documentation, such as clinical or legal documents. The accuracy, robustness, consistency, and faithfulness of the reasoning performed by the system is critical in this context, and it is important to carefully consider the domain-specific terminology, to handle linguistic constructs such as temporal associations or negations, and to have robustness to different writing styles and vocabularies."

    task_description = "Draft a summary for a natural messenger-like dialogue between two or more individuals. Attend to the most relevant information from the conversation, identifying the the names of the speakers and the actions and/or topics being discussed. Be thorough, accurate, clear, and brief, presenting the summary as a short paragraph written in the third-person point of view."

    dialog_description = "The dialogue exchanged via a messenger app is shown next."

    answer_description = "Compose the summary in a clear and consise language, presenting the important pieces of information in the third person."

    prompt_example_without_dialogue_and_special_chars = system_prompt + "\n" + task_description + "\n" + dialog_description + "\n" + answer_description + "\n"

    tokenized_content = tokenizer(prompt_example_without_dialogue_and_special_chars, return_tensors="pt", return_attention_mask=False)

    return len(tokenized_content['input_ids'][0])


def get_stats_from_numpy_array(numpy_array):
    print("size = {}".format(np.size(numpy_array)))
    print("mean = {}".format(np.mean(numpy_array)))
    print("std = {}".format(np.std(numpy_array)))
    print("25th percentile = {}".format(np.percentile(numpy_array, 25)))
    print("75th percentile = {}".format(np.percentile(numpy_array, 75)))
    print("max = {}".format(np.max(numpy_array)))
    print("min = {}".format(np.min(numpy_array)))
    print("")


### key_name = "summary"

key_name = "summary"

print("--- key_name = {}\n\n".format(key_name))

# stats n tokens -- train set
print("\npartition = train\n")

n_tokens_list = get_n_tokens_for_data(
    data=train_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))

# stats n tokens -- val set
print("\npartition = val\n")

n_tokens_list = get_n_tokens_for_data(
    data=val_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))

# stats n tokens -- test set
print("\npartition = test\n")

n_tokens_list = get_n_tokens_for_data(
    data=test_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))


### key_name = "dialogue"

key_name = "dialogue"

print("--- key_name = {}\n\n".format(key_name))

# stats n tokens -- train set
print("\npartition = train\n")

n_tokens_list = get_n_tokens_for_data(
    data=train_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))

# stats n tokens -- val set
print("\npartition = val\n")

n_tokens_list = get_n_tokens_for_data(
    data=val_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))

# stats n tokens -- test set
print("\npartition = test\n")

n_tokens_list = get_n_tokens_for_data(
    data=test_samsum_data,
    tokenizer=tokenizer,
    key_name=key_name
)

get_stats_from_numpy_array(np.array(n_tokens_list))

print("\n\nn tokens prompt example (without dialogue and special chars) = {}".format(str(get_n_tokens_for_prompt_example())))
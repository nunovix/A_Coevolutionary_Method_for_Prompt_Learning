import os
import shutil
from evo_functions import save_population

def load_population(iteration, root_folder, task = 'SemEval'):
    # Initialize the population dictionary
    population = {
        'prompts_dict': {},
        'history': {},
        'eval': [],
        'task': '',
        'prompts': [],
        'f1_scores': [],
        'confusion_matrix': [],
    }
    
    # Define the iteration folder path
    iteration_folder = os.path.join(root_folder, f"Iteration_{iteration}")
    
    # Read prompts_dict
    for filename in os.listdir(iteration_folder):
        if filename.endswith(".txt") and not filename.startswith("history_") and filename not in ["evaluations.txt", "f1_scores.txt", "confusion_matrix.txt", "full_eval.txt", "population.txt", "keep_list.txt"]:
            key = filename[:-4]  # Remove the .txt extension
            file_path = os.path.join(iteration_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = [line.split("->", 1)[1].strip() for line in lines if "->" in line]
            population['prompts_dict'][key] = values
    
    # Read history
    for filename in os.listdir(iteration_folder):
        if filename.startswith("history_") and filename.endswith(".txt"):
            key = filename[len("history_"):-4]
            file_path = os.path.join(iteration_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = [line.split("->", 1)[1].strip() for line in lines if "->" in line]
            population['history'][key] = values
    
    # Read evaluations
    additional_file_path = os.path.join(iteration_folder, "evaluations.txt")
    with open(additional_file_path, 'r') as file:
        population['eval'] = [line.strip() for line in file.readlines()]

    # Check if task-specific files exist and read them
    additional_file_path = os.path.join(iteration_folder, "f1_scores.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['f1_scores'] = [line.strip() for line in file.readlines()]

    additional_file_path = os.path.join(iteration_folder, "confusion_matrix.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['confusion_matrix'] = [line.strip() for line in file.readlines()]

    additional_file_path = os.path.join(iteration_folder, "full_eval.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['full_eval'] = [line.strip() for line in file.readlines()]

    # Read population
    additional_file_path = os.path.join(iteration_folder, "population.txt")
    with open(additional_file_path, 'r') as file:
        population_lines = [line.strip().split(", ") for line in file.readlines()]
        population['prompts'] = [line[0] for line in population_lines]
        # Populate eval without duplicating the same eval items from evaluations.txt
        eval_items = [line[1] for line in population_lines]
        population['eval'] = eval_items

    # Read keep_list
    keep_list = []
    additional_file_path = os.path.join(iteration_folder, "keep_list.txt")
    with open(additional_file_path, 'r') as file:
        keep_list = [line.strip() for line in file.readlines()]

    population['task'] = task

    return population, keep_list


# Sample data structure
iteration = 1
root_folder = "test_data"
population = {
    'prompts_dict': {'key1': ['value1', 'value2'], 'key2': ['value3']},
    'history': {'key1': ['value1', 'value2'], 'key2': ['value3']},
    'eval': ['eval1', 'eval2', 'eval3'],
    'task': 'SemEval',
    'prompts': ['prompt1', 'prompt2', 'prompt3'],
    'f1_scores': ['0.9', '0.8', '0.7'],
    'confusion_matrix': ['matrix1', 'matrix2', 'matrix3'],
}
keep_list = ['keep1', 'keep2']

# Save the sample data
save_population(iteration, population, root_folder, keep_list)

# Load the data back
loaded_population, loaded_keep_list = load_population(iteration, root_folder)


print(f"population-->{population}")
print(f"loaded_population-->{loaded_population}")

# Compare the original and loaded data
assert population == loaded_population, "Population data does not match!"
assert keep_list == loaded_keep_list, "Keep list does not match!"

print("Data successfully saved and loaded. The data structures match!")

# Cleanup
shutil.rmtree(root_folder)

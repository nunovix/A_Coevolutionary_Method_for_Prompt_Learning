import os
import shutil
import ast
from evo_functions import save_population, extract_lines_to_dict, pop_selection

def load_population(iteration, root_folder, task):
    # Initialize the population dictionary
    population = {
        'prompts_dict': {},
        'history': {},
        'eval': [],
        'full_eval': [],
        'task': task,
        'prompts': [],
        #'f1_scores': [],
        #'confusion_matrix': [],
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
                values = []
                for line in lines:
                    line = line.strip()
                    if '->' in line:
                        value = line.split('->', 1)[1].strip()
                        if value:
                            values.append(value)
                population['prompts_dict'][key] = values

    # Read history
    for filename in os.listdir(iteration_folder):
        if filename.startswith("history_") and filename.endswith(".txt"):
            key = filename[len("history_"):-4]
            file_path = os.path.join(iteration_folder, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
                values = []
                for line in lines:
                    line = line.strip()
                    if '->' in line:
                        value = line.split('->', 1)[1].strip()
                        if value:
                            values.append(value)
                population['history'][key] = values

    # Read evaluations
    additional_file_path = os.path.join(iteration_folder, "evaluations.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['eval'] = [line.strip() for line in file if line.strip()]

    # Read task-specific files
    additional_file_path = os.path.join(iteration_folder, "full_eval.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['full_eval'] = [line.strip() for line in file if line.strip()]

    """# Read task-specific files
    additional_file_path = os.path.join(iteration_folder, "f1_scores.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['f1_scores'] = [line.strip() for line in file if line.strip()]

    additional_file_path = os.path.join(iteration_folder, "confusion_matrix.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            population['confusion_matrix'] = [line.strip() for line in file if line.strip()]"""

    # Read population
    additional_file_path = os.path.join(iteration_folder, "population.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            lines = file.readlines()
            prompts = []
            evals = []
            for line in lines:
                line = line.strip()
                if ", " in line:
                    parts = line.rsplit(", ", 1)
                    if len(parts) == 2:
                        prompts.append(ast.literal_eval(parts[0]))
                        evals.append(float(parts[1]))
            population['prompts'] = prompts
            population['eval'] = evals

    # Read keep_list
    additional_file_path = os.path.join(iteration_folder, "keep_list.txt")
    if os.path.exists(additional_file_path):
        with open(additional_file_path, 'r') as file:
            keep_list = [line.strip() for line in file if line.strip()]
    else:
        keep_list = []

    return population, keep_list


"""
# Sample data structure
iteration = 1
root_folder = "test_data"
population = {
    'prompts_dict': {'key1': ['value1', 'value2'], 'key2': ['value3', 'value4']},
    'history': {'key1': ['histo value1', 'histo value2'], 'key2': ['histo value3', 'hidto vale 4']},
    'eval': [0.9, 0.8, 0.7],
    'full_eval': ["0.9", "0.8", "0.7"],
    'task': 'SemEval',
    'prompts': [{'key1': 0, 'key2': 0}, {'key1': 1, 'key2': 1}, {'key1': 1, 'key2': 0}],
    'f1_scores': ['0.9', '0.8', '0.7'],
    'confusion_matrix': ['matrix1', 'matrix2', 'matrix3'],
}
keep_list = ['keep1', 'keep2']

# Save the sample data
save_population(iteration, population, root_folder, keep_list)

# Load the data back
loaded_population, loaded_keep_list = load_population(iteration, root_folder, task='SemEval')

print(f"population-->{population}")
print(f"loaded_population-->{loaded_population}\n\n\n")

for key in loaded_population:
    print(f"population[{key}]-->{population[key]}")
    print(f"loaded_population[{key}]-->{loaded_population[key]}")
    print(f"equality-->{population[key] == loaded_population[key]}")

# Compare the original and loaded data
#assert population == loaded_population, "Population data does not match!"
#assert keep_list == loaded_keep_list, "Keep list does not match!"

#print("Data successfully saved and loaded. The data structures match!")

# Cleanup
shutil.rmtree(root_folder)
"""

def extract_max_eval_and_patience(root_folder, task):
    # Get all iteration folders including the initial one
    iteration_folders = [f for f in os.listdir(root_folder) if f.startswith("Iteration_")]
    iteration_folders.sort(key=lambda x: (int(x.split('_')[1]) if x != "Iteration_initial" else -1))
    
    max_eval_values = []
    max_eval_iteration = None
    max_eval_value = -float('inf')
    
    # Determine the iteration with the maximum evaluation value
    for iteration_folder in iteration_folders:
        evaluations_file_path = os.path.join(root_folder, iteration_folder, "evaluations.txt")
        
        if not os.path.exists(evaluations_file_path):
            continue
        
        with open(evaluations_file_path, 'r') as file:
            eval_values = [float(line.strip()) for line in file.readlines()]
        
        if eval_values:
            max_eval = max(eval_values)
            max_eval_values.append(max_eval)
            
            if max_eval > max_eval_value:
                max_eval_value = max_eval
                max_eval_iteration = iteration_folder
    
    # Determine the current iteration number and folder
    current_iteration_folder = iteration_folders[-1] if iteration_folders else "None"
    current_iteration_num = (int(current_iteration_folder.split('_')[1]) if current_iteration_folder != "Iteration_initial" else 0)
    
    # Calculate patience
    if max_eval_iteration:
        max_eval_iteration_num = int(max_eval_iteration.split('_')[1]) if max_eval_iteration != "Iteration_initial" else 0
        patience = 0
        for iteration_folder in iteration_folders:
            iteration_num = int(iteration_folder.split('_')[1]) if iteration_folder != "Iteration_initial" else 0
            if iteration_num > max_eval_iteration_num:
                evaluations_file_path = os.path.join(root_folder, iteration_folder, "evaluations.txt")
                
                if not os.path.exists(evaluations_file_path):
                    continue
                
                with open(evaluations_file_path, 'r') as file:
                    eval_values = [float(line.strip()) for line in file.readlines()]
                
                if eval_values:
                    max_eval = max(eval_values)
                    if max_eval <= max_eval_value:
                        patience += 1
                    else:
                        patience = 0
    
    # Load the population of the latest iteration
    print(f"current_iteration_num.: {current_iteration_num}")
    population, _ = load_population(current_iteration_num, root_folder, task)

    print(f"best iter no.: {max_eval_iteration_num}")
    #best_population, _ = load_population(max_eval_iteration_num, root_folder, task)
    #best_pop, _ = pop_selection(best_population, 1, 1)

    print(f"max_eval_values: {max_eval_values}")
    print(f"patience: {patience}")
    
    return max_eval_values, patience, population, current_iteration_num, best_pop


# Example usage
#root_folder = "RUNS_alg_2/SemEval_whighFalse_wselfTrue/Runs_2024-07-26_17-40-27_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue"
#max_eval_values, patience, population, current_iteration_num, best_pop = extract_max_eval_and_patience(root_folder, task='SemEval')

root_folder = "RUNS_alg_2/SemEval_whighFalse_wselfTrue/Runs_2024-07-26_17-40-27_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue"
#root_folder =  'RUNS_alg_2/MEDIQASUM/Runs_2024-08-01_17-24-23_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue'


max_eval_values, patience, population, current_iteration_num, best_pop = extract_max_eval_and_patience(root_folder, task='MEDIQASUM')

print("Max eval values for each iteration:", max_eval_values)
print("Patience value (iterations with no improvement):", patience)
print("Current iteration number:", current_iteration_num)
#print("Current population:", population)
for key in population:
    print(f"len(population[{key}])-->{len(population[key])}")
        
for dict in population['prompts_dict']:
    print(dict)
    print(f"len(population['prompts_dict'][dict])-->{len(population['prompts_dict'][dict])}")

prompts_path = 'INITIAL_PROMPTS/MEDIQASUM'

initial_prompts = extract_lines_to_dict(prompts_path, 
                                                       task = 'MEDIQASUM', 
                                                       task_w_one_shot=True,
                                                       )

for key in initial_prompts:
    print(f"key-->{key}")
    print(f"initial_prompts[key][0]-->{initial_prompts[key][0]}") 
    print(f"len(initial_prompts[key])-->{len(initial_prompts[key])}")  
    n_sub =  len(initial_prompts[key])

print(f"n_sub-->{n_sub}")

tam = []
for key in initial_prompts:
    tam.append(len(initial_prompts[key]))

print(f"tam[0]-->{tam[0]}")

print(f"best_pop-->{best_pop}")

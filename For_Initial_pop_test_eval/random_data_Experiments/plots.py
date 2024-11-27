import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, to_hex
import os

# function that creates and saves the plots for the iterations evolution
def plot_and_save_scores(all_scores, max_scores, directory_path, display_only_top_values, iteration_folders, 
                         y_min, y_max, score, keep_list):
    # Prepare the x-axis values (iteration numbers)
    x_values = range(len(all_scores))
    # Custom x-axis labels from the iteration folder names
    x_labels = [folder.replace('Iteration_', '') for folder in iteration_folders]

    # Create the plot
    plt.figure(figsize=(10, 6))

    if display_only_top_values:
        # Plot filename for only top scores
        plot_filename = 'top_scores_plot.png'
        # Plot only the maximum scores for each iteration
        plt.plot(x_values, max_scores, '-o', color='darkblue', label='Top Scores')
    
    else:
        # Plot filename for all scores
        plot_filename = 'all_scores_plot.png'
        # Define a list of base colors for the iterations
        base_colors = plt.cm.get_cmap('tab20', len(all_scores))

        # Plot all scores with lighter color for those not in keep_list
        for i, (scores, keep_indices) in enumerate(zip(all_scores, keep_list)):
            base_color = base_colors(i)  # Get the base color for this iteration
            # Convert base color to RGBA and then lighten the color for non-highlighted points
            lighter_color = to_rgba(base_color, alpha=0.12)  # Adjust alpha to make lighter
            # Plot all scores in lighter color
            plt.scatter([i] * len(scores), scores, color=lighter_color, label='Iteration {}'.format(i) if i == 0 else "")
            # Overlay highlighted scores in original color
            highlighted_scores = [scores[idx] for idx in keep_indices if idx < len(scores)]
            plt.scatter([i] * len(highlighted_scores), highlighted_scores, color=base_color)

        # Plot the line for top scores in a consistent color
        plt.plot(x_values, max_scores, '-o', color='black', label='Top Scores')

    # Labeling the plot
    plt.title('Scores by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel(score)
    plt.xticks(x_values, x_labels, rotation='vertical')  # Set custom x-axis labels
    plt.ylim(y_min, y_max)  # Set the y-axis range
    #plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Save the plot to the specified directory
    plt.savefig(os.path.join(directory_path, plot_filename))
    plt.close()  # Close the plot to free up memory
    
    print("Plots have been saved to:", directory_path)

# function that takes file path to folder with runs, created during the evolution of the model
# also takes y axix scale
# and label for the y axis, which is the score
def create_plots_from_RUNS_folder(directory_path):
    if "SemEval" in directory_path or "hyper" in directory_path:
        ymin = 0.50
        ymax = 0.80
        score = 'F1-Score'
    elif 'CSQA' in directory_path:
        ymin = 0.50
        ymax = 0.80
        score = 'Accuracy'
    elif 'ContractNLI' in directory_path:
        ymin = 0.40
        ymax = 1.00
        score = 'Accuracy'
    elif 'MEDIQASUM' in directory_path:
        ymin = 0.25
        ymax = 0.55
        score = 'Rouge-1 F1'
    elif 'LegalSumTOSDR' in directory_path:
        ymin = 0.00
        ymax = 0.50
        score = 'R1'
    else:
        print(f"Incorrect task name")
        return None
    
    # List all items in the directory
    items = os.listdir(directory_path)

    # Filter out items that are not directories or are 'Iteration_best'
    iteration_folders = [item for item in items 
                        if os.path.isdir(os.path.join(directory_path, item)) and item != 'Iteration_best']

    # Custom sorting function
    def custom_sort(folder_name):
        if folder_name == 'Iteration_initial':
            return -1  # Ensure 'Iteration_initial' comes first
        else:
            # Extract the iteration number and convert it to an integer for proper numerical sorting
            num_part = folder_name.split('_')[-1]
            return int(num_part) if num_part.isdigit() else float('inf')  # Non-numeric suffixes go at the end

    # Sort the folders using the custom function
    iteration_folders.sort(key=custom_sort)

    # Initialize a list to hold all scores lists and a list for max scores
    all_scores = []
    max_scores = []
    keep_lists = []

    for folder in iteration_folders:
        # Construct the path to the evaluation.txt file
        file_path = os.path.join(directory_path, folder, 'evaluations.txt')
        
        # Initialize a list to hold scores for this iteration
        scores = []
        
        # Open the file and read the scores
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a float and append to the scores list
                scores.append(float(line.strip()))
        
        # Append this iteration's scores to the all_scores list
        all_scores.append(scores)
        
        # Find and append the max score for this iteration to the max_scores list
        max_scores.append(max(scores))

        file_path = os.path.join(directory_path, folder, 'keep_list.txt')
        keep_list = []
        # Open the file and read the scores
        with open(file_path, 'r') as file:
            for line in file:
                # Convert each line to a float and append to the scores list
                keep_list.append(int(line.strip()))

        keep_lists.append(keep_list)

    plot_and_save_scores(all_scores, max_scores, directory_path, False, iteration_folders, ymin, ymax, score, keep_lists)  # For all scores
    plot_and_save_scores(all_scores, max_scores, directory_path, True, iteration_folders, ymin, ymax, score, keep_lists)  # For only top scores

    return None

path = "ContractNLI/retrieved_Runs_2024-07-21_12-18-30_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalse_new_evo_promptsTrue"
create_plots_from_RUNS_folder(path)

path = "ContractNLI/oracle_Runs_2024-09-18_00-36-00_N25_cp0.25_mp0.25_sampT10.0_fixed_evoFalsedq_dataFalse_reverseFalse_dev_ratioFalse_614"
create_plots_from_RUNS_folder(path)

path = "ContractNLI/baseline_Runs_2024-07-17_02-33-36_N25_cp0.0_mp0.5_sampTNone_fixed_evoTrue_new_evo_promptsTrue"
create_plots_from_RUNS_folder(path)



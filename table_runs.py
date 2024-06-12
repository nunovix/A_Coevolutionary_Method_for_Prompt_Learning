import os
import re
import shutil
import pandas as pd

# Path to the SemEval directory
semeval_path = 'RUNS_fine_tuning/SemEval_whighFalse_wselfFalse'
output_dir = 'hyperparameter_optimization_results_N10'
images_dir = os.path.join(output_dir, 'images')

# Create output directories if they don't exist
os.makedirs(images_dir, exist_ok=True)

# Regex pattern to extract hyperparameters from folder names
pattern = re.compile(r'Runs_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_N(\d+)_cp([0-9.]+)_mp([0-9.]+)_sampT([0-9.]+)')

# List to store the extracted information
data = []

# Function to get the maximum value from a file
def get_max_value_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            values = [float(line.strip()) for line in file.readlines()]
            return max(values)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to get the first value from a file
def get_first_value_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line = file.readline().strip()
            value = float(line.split('[')[1].split(']')[0])
            return value
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Function to get faithfulness and consistency from a file
def get_faithfulness_and_consistency(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            faithfulness = None
            consistency = None
            for line in lines:
                if line.startswith('Faithfulness:'):
                    faithfulness = float(line.split(':')[1].strip())
                if line.startswith('Consistency:'):
                    consistency = float(line.split(':')[1].strip())
            return faithfulness, consistency
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

# Function to count the number of improvements
def count_improvements(file_path):
    try:
        with open(file_path, 'r') as file:
            values = [float(line.strip()) for line in file.readlines()]
            improvements = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
            return improvements
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Iterate through the directories in the SemEval folder
for run_dir in os.listdir(semeval_path):
    match = pattern.match(run_dir)
    if match:
        # Extract hyperparameters
        N = int(match.group(1))
        cp = float(match.group(2))
        mp = float(match.group(3))
        sampT = float(match.group(4))
        
        # Paths for the scores and image
        run_dir_path = os.path.join(semeval_path, run_dir)
        initial_f1_path = os.path.join(run_dir_path, 'Iteration_initial', 'evaluations.txt')
        best_f1_path = os.path.join(run_dir_path, 'Iteration_best', 'evaluations.txt')
        test_f1_path = os.path.join(run_dir_path, 'Iteration_best', 'test_evaluation.txt')
        report_path = os.path.join(run_dir_path, 'Iteration_best', 'test_report.txt')
        image_path = os.path.join(run_dir_path, 'all_scores_plot.png')
        scores_evo_path = os.path.join(run_dir_path, 'scores_evo.txt')
        
        # Get scores
        initial_f1 = get_max_value_from_file(initial_f1_path)
        best_f1 = get_max_value_from_file(best_f1_path)
        test_f1 = get_first_value_from_file(test_f1_path)
        faithfulness, consistency = get_faithfulness_and_consistency(report_path)
        
        # Count the number of improvements
        num_improvements = count_improvements(scores_evo_path)
        
        # Check if the image file exists
        if os.path.exists(image_path):
            # Copy the image to the output directory
            new_image_path = os.path.join(images_dir, f'{run_dir}.png')
            shutil.copy(image_path, new_image_path)
            image_relative_path = os.path.relpath(new_image_path, output_dir)
            print(f"Copied image: {image_path} to {new_image_path}")
        else:
            image_relative_path = ''  # Clear the image path if the file doesn't exist
            print(f"Image not found: {image_path}")
        
        # Append extracted data to the list
        data.append({
            'Sampling Temperature': sampT,
            'Crossover Probability': cp,
            'Mutation Probability': mp,
            'Population Size': N,
            'Dev Set Initial F1 Score': initial_f1,
            'Dev Set Best F1 Score': best_f1,
            'Test Set F1 Score': test_f1,
            'Test Set Faithfulness': faithfulness,
            'Test Set Consistency': consistency,
            'Evolution of F1-score in the Dev Set': image_relative_path,
            'Number of Improvements during the evolution': num_improvements  # Use calculated value
        })

# Create a DataFrame with the updated data
df = pd.DataFrame(data)

# Group by Sampling Temperature, Crossover Probability, and Mutation Probability
df_sorted = df.sort_values(by=['Sampling Temperature', 'Crossover Probability', 'Mutation Probability'])

# Format numerical values to 4 decimal points
df_sorted = df_sorted.round(4)

# Function to render the table with images
def render_html_table(df, title):
    html = f'''
    <html>
    <head>
        <title>{title}</title>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid black;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                display: block;
                margin-left: auto;
                margin-right: auto;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <table>
    '''
    html += '<tr>' + ''.join([f'<th>{col}</th>' for col in df.columns if col != 'run_dir']) + '</tr>'
    for _, row in df.iterrows():
        html += '<tr>'
        for col in df.columns:
            if col == 'Image':
                if row[col]:
                    html += f'<td><img src="{row[col]}" width="300"></td>'
                else:
                    html += '<td>No image</td>'
            elif col != 'run_dir':
                html += f'<td>{row[col]}</td>'
        html += '</tr>'
    html += '''
        </table>
    </body>
    </html>
    '''
    return html

# Save the HTML table views
html_views = [
    ("hyperparameter_optimization_by_sampling_temperature.html", df_sorted.sort_values(by=['Sampling Temperature'])),
    ("hyperparameter_optimization_by_crossover_probability.html", df_sorted.sort_values(by=['Crossover Probability'])),
    ("hyperparameter_optimization_by_mutation_probability.html", df_sorted.sort_values(by=['Mutation Probability']))
]

for filename, dataframe in html_views:
    html_table = render_html_table(dataframe, filename.replace('.html', '').replace('_', ' ').title())
    html_output_path = os.path.join(output_dir, filename)
    with open(html_output_path, 'w') as file:
        file.write(html_table)
    print(f"HTML table saved as '{html_output_path}'")

# Calculate averages for each hyperparameter
numeric_columns = ['Population Size', 'Dev Set Initial F1 Score', 'Dev Set Best F1 Score', 'Test Set F1 Score', 'Test Set Faithfulness', 'Test Set Consistency', 'Number of Improvements during the evolution']
averages = {
    'Sampling Temperature': df.groupby('Sampling Temperature')[numeric_columns].mean().round(4),
    'Crossover Probability': df.groupby('Crossover Probability')[numeric_columns].mean().round(4),
    'Mutation Probability': df.groupby('Mutation Probability')[numeric_columns].mean().round(4)
}

# Save the summary tables
for key, dataframe in averages.items():
    html_table = render_html_table(dataframe.reset_index(), f"Average Results by {key}")
    html_output_path = os.path.join(output_dir, f"average_results_by_{key.lower().replace(' ', '_')}.html")
    with open(html_output_path, 'w') as file:
        file.write(html_table)
    print(f"Summary HTML table saved as '{html_output_path}'")

print("All HTML tables and summary views have been saved successfully.")
# Print a sample HTML output
sample_html = render_html_table(df_sorted.head(5), "Sample HTML Output")
print(sample_html)


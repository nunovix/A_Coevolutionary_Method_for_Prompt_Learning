#import evo_functions as evo
from evo_functions import extract_lines_to_dict, extract_SemEval_data, prompt_creation_semeval_self
import os
from collections import Counter

# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

folder_path = 'SemEval_self_initial_population_prompts'
initial_population_prompts = extract_lines_to_dict(folder_path)
#evolutionary_prompts = evo.extract_lines_to_dict("evolutionary_prompts")
# evaluate test file from semeval or another task
data_expanded = extract_SemEval_data(type = 'dev')# evaluate test file from semeval or another task

print(data_expanded[0].keys())
print('\n\n')

label = data_expanded[0]['label']

"""text = data_expanded[0]['text']
statement = data_expanded[0]['statement']
spans = data_expanded[0]['spans']
spans_index = data_expanded[0]['spans_index']"""

print(f"initial_population_prompts.keys()-->{initial_population_prompts.keys()}\n\n")

statement = initial_population_prompts['statement_description'][0]
ctr =initial_population_prompts['ctr_description'][0]
answer = initial_population_prompts['answer_description'][0]
task  = initial_population_prompts['task_description'][0]

self_A  = initial_population_prompts['self_A'][0]
self_B  = initial_population_prompts['self_B'][0]
self_C  = initial_population_prompts['self_C'][0]

print(f"\nstatement-->{statement}\n")
print(f"ctr-->{ctr}\n")
print(f"answer-->{answer}\n")
print(f"task-->{task}\n")

print(f"self_A-->{self_A}\n")
print(f"self_B-->{self_B}\n")
print(f"self_C-->{self_C}\n")

prompts = prompt_creation_semeval_self(data_expanded, task_description=task, ctr_description=ctr, statement_description=statement, answer_description=answer)

labels = []
for prompt in prompts:
    #print(f"\n\n\n\n\n\n\ntext-->{prompt['text']}")
    labels.append(prompt['label'])

print(f"counter-->{Counter(labels)}")
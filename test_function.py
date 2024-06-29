# function to remove duplicate and remap
def remove_duplicates_and_remap(population):
    prompt_dict = population['prompts_dict']
    prompts = population['prompts']
    # Step 1: Create a mapping of valid indices and update prompt_dict
    remap_dict = {}
    for part, possibilities in prompt_dict.items():
        seen = {}
        new_possibilities = []
        new_index = 0
        index_mapping = {}
        
        for old_index, possibility in enumerate(possibilities):
            if possibility not in seen:
                seen[possibility] = new_index
                new_possibilities.append(possibility)
                index_mapping[old_index] = new_index
                new_index += 1
            else:
                index_mapping[old_index] = seen[possibility]
        
        prompt_dict[part] = new_possibilities
        remap_dict[part] = index_mapping

    # Step 2: Update prompts
    for prompt in prompts:
        for part, old_index in list(prompt.items()):
            if part in remap_dict:
                prompt[part] = remap_dict[part][old_index]

    population['prompts_dict'] = prompt_dict
    population['prompts'] = prompts

    return population

pop_1 = {# Test Case 1: Basic test case with some duplicates
'prompts_dict' : {
    'part_A': ['possibility_1', 'possibility_2', 'possibility_1'],
    'part_B': ['possibility_3', 'possibility_3']
},
'prompts' : [
    {'part_A': 0, 'part_B': 1},
    {'part_A': 2, 'part_B': 0}
]}

pop_2 = {# Test Case 1: Basic test case with some duplicates
'prompts_dict' : {
    'part_X': ['possibility_4', 'possibility_5'],
    'part_Y': ['possibility_6']
},
'prompts' : [
    {'part_X': 0, 'part_Y': 0},
    {'part_X': 1, 'part_Y': 0}
]}

pop_3 = {# Test Case 1: Basic test case with some duplicates
'prompts_dict' : {
    'part_A': ['possibility_1', 'possibility_1', 'possibility_1'],
    'part_B': ['possibility_2', 'possibility_2', 'possibility_2']
},
'prompts' : [
    {'part_A': 0, 'part_B': 1},
    {'part_A': 2, 'part_B': 0}
]}

pop_4 = {# Test Case 1: Basic test case with some duplicates
'prompts_dict' : {
    'part_A': ['possibility_1', 'possibility_2', 'possibility_1', 'possibility_3'],
    'part_B': ['possibility_3', 'possibility_2', 'possibility_3']
},
'prompts' : [
    {'part_A': 0, 'part_B': 1},
    {'part_A': 2, 'part_B': 2}
]}

# Applying function to test cases
print("Test Case 1")
pop_1 = remove_duplicates_and_remap(pop_1)
print(pop_1['prompts_dict'])
print(pop_1['prompts'])

print("Test Case 2")
pop_2 = remove_duplicates_and_remap(pop_2)
print(pop_2['prompts_dict'])
print(pop_2['prompts'])

print("Test Case 3")
pop_3 = remove_duplicates_and_remap(pop_3)
print(pop_3['prompts_dict'])
print(pop_3['prompts'])

print("Test Case 4")
pop_4 = remove_duplicates_and_remap(pop_4)
print(pop_4['prompts_dict'])
print(pop_4['prompts'])

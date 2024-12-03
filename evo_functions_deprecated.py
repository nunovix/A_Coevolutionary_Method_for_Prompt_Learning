# functions that were previously developed in evo_functions.py, 
# but were then not used/evolved/fused into the evo_alg_2 function
# from said file

#########################################################################################################

# !!!!!!! DEPRECATED since you updated the representation of population['prompt'] from a list of list to a lsit of dictionaries
# function to run the evolutionary alg, with a initial population of prompts
# evolutionary prompts (1 for mutation, 1 for combination)
# hf model and tokenizer
# hyperparameters of the algorithm
def evo_alg(task, initial_prompts, evolutionary_prompts,
            model_name = "mistralai/Mistral-7B-Instruct-v0.2",
            quantize_model_4bits = True,
            n_pop = 5, # initial population size and the number of elements kepts at each iteration
            n_keep = 5,
            n_top = 5, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
            n_combinations = 15,
            patience = 10,
            max_iter = 50,
            temperature = 1.0, #temperature for decoding combined and mutated
            top_p = 0.8, #sampling for decoding combined and mutated
            save = True,
            eval_data = 'dev', # dev or train
            data_size = 0): # no. of samples where the prompts are evaluated, if =0 all are used

    if task != 'SemEval' and task != 'SemEval_self' and task != 'CSQA' and task != 'ContractNLI':
        print(f"Incorrect task selected")
        return None
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
        
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == True:
        n_pop = tam[0]
    else:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # check n_top
    if n_top > n_pop:
        n_top = n_pop

    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(task, tokenizer)
    
    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task)
        print(f"Root folder created: {root_folder}")

    if task == 'SemEval' or task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = 'dev')

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)
    
    patience_counter = 0
    iter = 0

    initial_population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size)

    print(f"initial_population eval-->{initial_population['eval']}")
    best_score_iterations.append(max(initial_population['eval']))
    
    if save == True:
        save_population('initial', initial_population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 
        mutated_prompts = {key: [] for key in initial_population['prompts_dict'].keys()}
        combined_prompts = {key: [] for key in initial_population['prompts_dict'].keys()}
        mutated_history = {key: [] for key in initial_population['prompts_dict'].keys()}
        combined_history = {key: [] for key in initial_population['prompts_dict'].keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop), desc = f"iteration {iter} - Mutating prompts"):
            # iterate through the subprompts
            for j in initial_population['prompts_dict'].keys():

                # mutate each subprompt and add to the mutated population prompts
                mutated = mutate_prompt(initial_population['prompts_dict'][j][i], evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 

                mutated_prompts[j].append(mutated)
                mutated_history[j].append(f"mutated from {i} at iteration {iter+1}")

        mutated_population = create_population(task, mutated_prompts, initial = False,
                                               data_expanded = data_expanded, 
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=mutated_history)

        #print(f"initial_population['eval']-->{initial_population['eval']}")
        population = combine_populations(initial_population, mutated_population)
        #print(f"population['eval']-->{population['eval']}")
        print(f"initial_population['eval']-->{initial_population['eval']}")

        for i in tqdm(range(n_combinations), desc = f"iteration {iter} - Combining prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            m = 0
            for j in population['prompts_dict'].keys():

                sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2)
                print(f"sel4comb-->{sel4comb}")

                # combine each subprompt randomly selected and add to the combined and total population
                combined = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][m]], population['prompts_dict'][j][population['prompts'][sel4comb[1]][m]],
                                           evolutionary_prompts['combination_prompts'][0], model, tokenizer)
                combined_prompts[j].append(combined)
                combined_history[j].append(f"combined from [{population['prompts'][sel4comb[0]][m]}] and [{population['prompts'][sel4comb[1]][m]}] at iteration {iter+1}")
                m+=1

        combined_population = create_population(task, combined_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history=combined_history)

        population = combine_populations(population, combined_population)

         # if improved patience returns to 0
        #print(f"before pat counter determination")
        #print(f"max(population['eval'])-->{max(population['eval'])}")
        #print(f"max(initial_population['eval'])-->{max(initial_population['eval'])}")
        #print(f"max(population['eval']) > max(initial_population['eval'])-->{max(population['eval']) > max(initial_population['eval'])}")
        if max(population['eval']) > max(initial_population['eval']):
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")

        keep_pop = deepcopy(sorted_population)

        # Create a new dictionary with the same keys, but values are lists with only the selected indices
        # how the population is being maintained
        n_pop = n_keep
        keep_pop, keep_list = pop_selection(keep_pop, n_pop, n_top) # !!!!!!!!!!

        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, keep_list)
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

        initial_population = deepcopy(keep_pop)
        print(f"evaluation of keepers for next gen-->{initial_population['eval']}")
        print(f"keep_list-->{keep_list}")

        print(f"patience_counter-->{patience_counter}")

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_pop, keep_list = pop_selection(sorted_population, 1, 1)
    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details(root_folder, n_pop, n_keep, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            n_combinations,
                            patience,
                            max_iter,
                            iter,
                            temperature,
                            top_p,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits)
            create_plots_from_RUNS_folder(root_folder)

    return best_pop, best_score_iterations




# varies from alg_2 by instead of performing crossovers followed by mutations performing either only a crossover or a mutation to 
# generate new individuals
##############################################################
def evo_alg_3(task,
              model_name = "microsoft/Phi-3-mini-128k-instruct",
              quantize_model_4bits = True,
              n_pop = 5, # initial population size and the number of elements kepts at each iteration
              n_top = 1, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
              operation_prob=0.75,
              mutation_operation_prob=0.5, # 1-mutation_operation_prob will be the crossover operation probability, considering we only have 2 operations
              sampling_T = 5.0,
              patience = 20,
              max_iter = 200,
              save = True,
              eval_data = 'dev', # dev or train
              data_size = 0, # no. of samples where the prompts are evaluated, if =0 all are used
              retrieve_examples = False, # use retrieval with embedding model instead of random for 1-shot learning
              task_w_self_reasoning = False,
              task_w_one_shot = False,
              task_w_highlight = False,
              do_test_eval = True,
              fixed_evo_prompts = True,
              new_evo_prompt_format = True,
              task_w_oracle_spans = True,
              ): 
    
    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    
    data_expanded, initial_prompts, evolutionary_prompts, trie, new_mutation_prompts, new_cross_prompts = sel_task_dataset_initial_prompts_evo_prompts(task_name=task,
                                                                                                                                                        tokenizer=tokenizer,
                                                                                                                                                        w_one_shot=task_w_one_shot,
                                                                                                                                                        w_self_reasoning=task_w_self_reasoning,
                                                                                                                                                        w_highlight=task_w_highlight
                                                                                                                                                        )
    
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task,
                                         alg = 'alg_3',
                                         operation_prob=operation_prob,
                                         mutation_operation_prob=mutation_operation_prob,
                                         N=n_pop,
                                         sampling_T=sampling_T,
                                         task_w_self_reasoning = task_w_self_reasoning,
                                         task_w_highlight = task_w_highlight,
                                         fixed_evo_prompts = fixed_evo_prompts,
                                         new_evo_prompts=new_evo_prompt_format
                                         )
        print(f"Root folder created: {root_folder}")

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, 
                                   initial_prompts, 
                                   initial = True,
                                   n_pop=n_pop,
                                   data_expanded = data_expanded, 
                                   model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                   task_w_one_shot = task_w_one_shot,
                                   task_w_highlight = task_w_highlight,
                                   task_w_self_reasoning = task_w_self_reasoning,
                                   task_w_oracle_spans=task_w_oracle_spans,)

    n_sub = len(population['prompts_dict'][list(population['prompts_dict'].keys())[0]])

    patience_counter = 0
    iter = 0
    #print(f"initial_population eval-->{population['eval']}")
    best_score_iterations.append(max(population['eval']))

    # for the best individual baseline related change
    best_pop, keep_list = pop_selection(population, 1, 1)
    
    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:

        # score best, done here so it can work as the baseline as well, as the best individual is not neecessarily passed to the next generation
        # Create a new dictionary with the same keys, but values are lists with only the selected indices
        if max(population['eval']) >= best_pop['eval'][0]: 
            best_pop, keep_list = pop_selection(population, 1, 1)

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_sub), desc = f"iteration {iter} - generating off springs prompts"):
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():

                #soft_max_scores = softmax(np.array(population['eval'])/sampling_T)
                #print(f"population['eval']-->{population['eval']}")
                soft_max_scores = softmax_samp_T(population['eval'], sampling_T)
                #print(f"soft_max_scores-->{soft_max_scores}")
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=2, replace=False, p = soft_max_scores)) 

                # apply an operation with a probability
                if random.random() <= operation_prob:
                    if random.random() <= 1 - mutation_operation_prob and population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]] != population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]]:
                        cross_prompt_index = {}
                        cross_prompt = {}
                        # wheter or not to randomly select mutation prompt from existing ones
                        if fixed_evo_prompts == False:

                            if new_evo_prompt_format == False:
                                cross_index = random.choice(list(range(len(evolutionary_prompts['combination_prompts']))))
                            else:
                                for key in new_cross_prompts:
                                    #print(f"key-->{key}")
                                    cross_prompt_index[key] = random.choice(list(range(len(new_cross_prompts[key]))))
                                    cross_prompt[key] = new_cross_prompts[key][cross_prompt_index[key]]
                            
                        else:
                            if new_evo_prompt_format == False:
                                cross_index = 0
                            else:
                                for key in new_cross_prompts:
                                    #print(f"key-->{key}")
                                    cross_prompt_index[key] = 0
                                    cross_prompt[key] = new_cross_prompts[key][cross_prompt_index[key]]

                        # combine each subprompt randomly selected and add to the combined and total population
                        if new_evo_prompt_format == False:
                            new_prompt = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], 
                                                        population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]],
                                                        evolutionary_prompts['combination_prompts'][cross_index], 
                                                        model, tokenizer)
                            hist = f"crossover between [{population['prompts'][sel4comb[0]][j]}] and [{population['prompts'][sel4comb[1]][j]}] using cross prompt {cross_index}"
                        else:
                            new_prompt = new_crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], 
                                                            population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]],
                                                            cross_prompt, 
                                                            model, tokenizer)
                            hist = f"crossover between [{population['prompts'][sel4comb[0]][j]}] and [{population['prompts'][sel4comb[1]][j]}] using cross prompt {cross_prompt_index}"

                    else: # apply mutation with prob mutation_operation_prob
                        mutation_prompt_index = {}
                        mutation_prompt = {}

                        # wheter or not to randomly select mutation prompt from existing ones
                        if fixed_evo_prompts == False:
                            if new_evo_prompt_format == False:
                                mut_index = random.choice(list(range(len(evolutionary_prompts['mutation_prompts']))))
                            else:
                                for key in new_mutation_prompts:
                                    #print(f"key-->{key}")
                                    mutation_prompt_index[key] = random.choice(list(range(len(new_mutation_prompts[key]))))
                                    mutation_prompt[key] = new_mutation_prompts[key][mutation_prompt_index[key]]

                        else:
                            if new_evo_prompt_format == False:
                                mut_index = 0
                            else:
                                for key in new_mutation_prompts:
                                    mutation_prompt_index[key] = 0
                                    mutation_prompt[key] = new_mutation_prompts[key][mutation_prompt_index[key]]

                        if new_evo_prompt_format == False:
                            new_prompt = mutate_prompt(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], 
                                                    evolutionary_prompts['mutation_prompts'][mut_index], 
                                                    model, tokenizer) 
                            hist=f"mutation using mut prompt {mut_index}"
                        else:
                            new_prompt = new_mutate_prompt(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]],
                                                        mutation_prompt,
                                                        model, tokenizer)
                            hist=f"mutation using mut prompt {mutation_prompt_index}"

                else: # nothing happens to the given subprompt, it is just a copy of the first chosen one. with prob 1-operation_prob
                    new_prompt = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                    hist = f"copy of [{population['prompts'][sel4comb[0]][j]}]"

                # adding subprompt and history, same for all cases
                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(new_prompt)
                offspring_history[j].append(hist)

        # after all the new prompts are generated create and evaluate new individuals
        offspring_population = create_population(task, 
                                                 offspring_prompts, 
                                                 initial = False,
                                                 n_pop = n_pop-n_top,
                                                 data_expanded = data_expanded, 
                                                 model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                 history = offspring_history, 
                                                 task_w_one_shot = task_w_one_shot,
                                                 task_w_highlight = task_w_highlight,
                                                 task_w_self_reasoning = task_w_self_reasoning,
                                                 task_w_oracle_spans=task_w_oracle_spans,
                                                 )

        if n_top == 0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)

        population = remove_duplicates_and_remap(population)

        if max(population['eval']) > best_pop['eval'][0]:
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population = sort_pop(population)
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")


        # Call the function
        if save == True:
            #print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    # best_pop, keep_list = pop_selection(sorted_population, 1, 1) # DEPRECATED

    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            operation_prob=operation_prob,
                            mutation_operation_prob=mutation_operation_prob,
                            retrieve_examples=retrieve_examples,
                            alg='alg_3',
                            )
            
            create_plots_from_RUNS_folder(root_folder)

    if do_test_eval == True:
        print(f"test set evaluation")
        test_eval(task=task, RUN_folder_path = root_folder, model_name=model_name)

    return best_pop, best_score_iterations





##############################################################
# baseline algorithm that performs a MC search
# only applies 
def alg_baseline(task, initial_prompts, evolutionary_prompts,
            model_name = "mistralai/Mistral-7B-Instruct-v0.2",
            quantize_model_4bits = True,
            n_pop = 5, # initial population size and the number of elements kepts at each iteration
            n_top = 0, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
            mutation_prob=0.5,
            patience = 10,
            max_iter = 50,
            temperature = 1.0, #temperature for decoding combined and mutated
            top_p = 0.8, #sampling for decoding combined and mutated
            save = True,
            eval_data = 'dev', # dev or train
            data_size = 0,
            test_eval = True ): # no. of samples where the prompts are evaluated, if =0 all are used

    if task != 'SemEval' and task != 'SemEval_self' and task != 'CSQA' and task != 'ContractNLI':
        print(f"Incorrect task selected")
        return None
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # load model and tokenizer
    # wether or not to quantize model
    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(task, tokenizer)

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()

    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task, alg = 'baseline')
        print(f"Root folder created: {root_folder}")

    if task == 'SemEval' or task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = eval_data)

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                           n_pop=n_pop,)

    # if needed increase initial population by generating mutated variations of the initial promtp list
    if n_pop > tam[0]:
        print(f"n_pop > tam[0]-->{n_pop} > {tam[0]}")

        extra_prompts = {key: [] for key in initial_prompts.keys()}
        extra_history = {key: [] for key in initial_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop-tam[0]), desc = f"Generating extra initial pop"):
            # iterate through the subprompts
            for j in initial_prompts.keys():
                # mutate each subprompt and add to the mutated population prompts
                #rndm choice of the operator to be used
                mutation_prompt_index = random.choice(list(range(len(evolutionary_prompts['mutation_prompts']))))
                #random choice of the prompt to be mutated
                ind_prompt = random.choice(list(range(tam[0])))
                mutated = mutate_prompt(initial_prompts[j][ind_prompt], evolutionary_prompts['mutation_prompts'][mutation_prompt_index], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                
                hist = f"mutated from {ind_prompt}, using mutation prompt {mutation_prompt_index}"

                extra_prompts[j].append(mutated)
                extra_history[j].append(hist)

        extra_population = create_population(task, extra_prompts, initial = True,
                                               data_expanded = data_expanded,
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=extra_history,
                                               n_pop=n_pop,)
        population = combine_populations(population, extra_population)

    patience_counter = 0
    iter = 0
    print(f"initial_population eval-->{population['eval']}")
    #print(f"len(population['eval'])-->{len(population['eval'])}")
    best_score_iterations.append(max(population['eval']))

    # pop with best individual at start
    best_pop, _ = pop_selection(population, 1, 1)
    best_iter = iter

    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_pop-n_top), desc = f"iteration {iter} - generating off springs prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():

                # random choice (NOT weighted)
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=1, replace=False)) #better way
                #print(f"sel4comb-->{sel4comb}")

                # apply mutation with probability crossover_prob, else off spring remains the same
                if random.random() <= mutation_prob:
                    mutated = mutate_prompt(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                    hist = f"mutation of [{population['prompts'][sel4comb[0]][j]}]"
                else:
                    mutated = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                    hist=f" copy of [{population['prompts'][sel4comb[0]][j]}]"

                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(mutated)
                offspring_history[j].append(hist)

        offspring_population = create_population(task, offspring_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history = offspring_history,
                                                n_pop=n_pop,)

        if n_top == 0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)


        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")

        # upgraded best one that is stored if improvement is shown
        new_best_pop, _ = pop_selection(sorted_population, 1, 1)
        if new_best_pop['eval'][0] > best_pop['eval'][0]:
            best_pop = deepcopy(new_best_pop)
            best_iter = iter+1
            patience_counter = 0
        else:
            patience_counter += 1

        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1


    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            alg='baseline',
                            best_iter=best_iter)
            
            create_plots_from_RUNS_folder(root_folder)

    if test_eval == True:
        try:
            print(f"test set evaluation")
            test_eval(task=task, RUN_folder_path = root_folder, model_name=model_name)
        except:
            print('error in the test set predictions')
            pass

    return best_pop, best_score_iterations





##############################################################

# hyperevolution algorith
# mutation pormpts are evaluatedd by generating N variations for a given set of subprompts
# the score will be the average across those N examples
def evo_alg_hyper(task,
                evaluation_task,
                initial_prompts, 
                hyper_evolutionary_prompts,
                model_name = "mistralai/Mistral-7B-Instruct-v0.2",
                quantize_model_4bits = True,
                n_pop = 5, # initial population size and the number of elements kepts at each iteration
                n_top = 1, # how many of the top performing options are being kept in the population at the end of each iteration (rest are randomized)
                mutation_prob=0.5,
                crossover_prob=0.5,
                sampling_T = 5.0,
                patience = 10,
                max_iter = 50,
                temperature = 1.0, #temperature for decoding combined and mutated
                top_p = 0.8, #sampling for decoding combined and mutated
                save = True,
                eval_data = 'dev', # dev or train
                data_size = 0, # no. of samples where the prompts are evaluated, if =0 all are used
                N = 10,
                eval_mutation_prob = 0.8):
    
    # check number of examples of each subprompts is the same
    tam = []
    for key in initial_prompts:
        tam.append(len(initial_prompts[key]))
    # Check if all elements are equal
    all_equal = all(element == tam[0] for element in tam)
    if all_equal == False:
        print(f"The no. of elements in each subprompt differs")
        return None, None
    
    # load model and tokenizer
    # wether or not to quantize model

    model, tokenizer = load_model(checkpoint = model_name, quantized = quantize_model_4bits)
    trie = get_Marisa_Trie(evaluation_task, tokenizer)

    # list to save best score at each iteration
    best_score_iterations = []
    start_time = datetime.now()
    
    # Call the function to create the folder and print its name
    if save == True:
        root_folder = create_root_folder(task, alg=task)
        print(f"Root folder created: {root_folder}")

    if evaluation_task == 'SemEval' or evaluation_task == 'SemEval_self':
        # extract SemEval data
        data_expanded = extract_SemEval_data(type = eval_data)
    elif evaluation_task == "CSQA":
        data_expanded = extract_CSQA_data(type = eval_data)
    elif evaluation_task == "ContractNLI":
        data_expanded = extract_ContractNLI_data(type = eval_data)

    if data_size == 0 or data_size > len(data_expanded):
        data_size = len(data_expanded)

    population = create_population(task, initial_prompts, initial = True,
                                           data_expanded = data_expanded, 
                                           model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                           N=N,
                                           mutation_prob=eval_mutation_prob)

    # if needed increase initial population by generating mutated variations of the initial promtp list
    if n_pop > tam[0]:
        print(f"n_pop > tam[0]-->{n_pop} > {tam[0]}")

        extra_prompts = {key: [] for key in initial_prompts.keys()}
        extra_history = {key: [] for key in initial_prompts.keys()}

        # iterate through each prompt to generate mutations
        for i in tqdm(range(n_pop-tam[0]), desc = f"Generating extra initial pop"):
            # iterate through the subprompts
            for j in initial_prompts.keys():
                # mutate each subprompt and add to the mutated population prompts
                #rndm choice of the operator to be used
                mutation_prompt_index = random.choice(list(range(len(hyper_evolutionary_prompts['mutation_prompts']))))
                #random choice of the prompt to be mutated
                ind_prompt = random.choice(list(range(tam[0])))
                mutated = mutate_prompt(initial_prompts[j][ind_prompt], hyper_evolutionary_prompts['mutation_prompts'][mutation_prompt_index], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                
                hist = f"mutated from {ind_prompt}, using hyper-mutation prompt {mutation_prompt_index}"

                extra_prompts[j].append(mutated)
                extra_history[j].append(hist)

        extra_population = create_population(task, extra_prompts, initial = True,
                                               data_expanded = data_expanded,
                                               model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                               history=extra_history,
                                               N=N,
                                               mutation_prob=eval_mutation_prob)
        
        population = combine_populations(population, extra_population)

    patience_counter = 0
    iter = 0
    print(f"initial_population eval-->{population['eval']}")
    #print(f"len(population['eval'])-->{len(population['eval'])}")
    best_score_iterations.append(max(population['eval']))
    
    if save == True:
        save_population('initial', population, root_folder, keep_list=list(range(n_pop)))
        print(f"Data saved for iteration {iter}.")
    
    while patience_counter < patience and iter < max_iter:
        
        # mutate population 

        offspring_prompts = {key: [] for key in population['prompts_dict'].keys()}
        offspring_history = {key: [] for key in population['prompts_dict'].keys()}

        best_score_at_start = max(population['eval'])
        # select elite population, n_top elements
        if n_top>0:
            elite_population, _ = pop_selection(population, n_top, n_top)

        for i in tqdm(range(n_pop-n_top), desc = f"iteration {iter} - generating off springs prompts"):
            #sel4comb = random.choices(range(len(population['eval'])), weights=population['eval'], k=2) # !!!!!!!!!!
            # iterate through the subprompts
            for j in population['prompts_dict'].keys():
                #print(f"len(population['eval'])-->{len(population['eval'])}")
                soft_max_scores = softmax(np.array(population['eval'])/sampling_T)
                #print(f"soft_max_scores-->{soft_max_scores}")
                sel4comb = list(np.random.choice(range(len(population['eval'])), size=2, replace=False, p = soft_max_scores)) #better way

                # apply crossover with probability crossover_prob, else off spring is copy of parent
                if random.random() <= crossover_prob:
                    # combine each subprompt randomly selected and add to the combined and total population
                    combined = crossover_prompts(population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]], population['prompts_dict'][j][population['prompts'][sel4comb[1]][j]],
                                           hyper_evolutionary_prompts['combination_prompts'][0], model, tokenizer,
                                           temperature=temperature, top_p=top_p)
                    hist = f"crossover between [{population['prompts'][sel4comb[0]][j]}] and [{population['prompts'][sel4comb[1]][j]}]"
                else:
                    combined = population['prompts_dict'][j][population['prompts'][sel4comb[0]][j]]
                    hist = f"copy of [{population['prompts'][sel4comb[0]][j]}]"
                
                # apply mutation with probability crossover_prob, else off spring remains the same
                if random.random() <= mutation_prob:
                    mutated = mutate_prompt(combined, hyper_evolutionary_prompts['mutation_prompts'][0], 
                                        model, tokenizer, temperature=temperature, top_p=top_p) 
                    hist+=f" followed by mutation"
                else:
                    mutated = combined
                    hist+=f" "

                hist+=f" from iteration {iter}"
                offspring_prompts[j].append(mutated)
                offspring_history[j].append(hist)

        offspring_population = create_population(task, offspring_prompts, initial = False,
                                                data_expanded = data_expanded, 
                                                model=model, tokenizer=tokenizer, trie=trie, n_samples = data_size,
                                                history = offspring_history,
                                                N=N,
                                                mutation_prob=eval_mutation_prob)

        if n_top ==0:
            population = deepcopy(offspring_population)
        else:
            population = combine_populations(elite_population, offspring_population)

        if max(population['eval']) > best_score_at_start:
            patience_counter = 0
        # difference to the if is that there was no overall improvment so patience counter increases
        else:
            patience_counter += 1

        sorted_population = sort_pop(population) # !!!!!!!!!!
        print(f"sorted evaluation at iteration {iter + 1} (all elements)-->{sorted_population['eval']}")


        # Call the function
        if save == True:
            print(f"sorted_population['prompts']-->{sorted_population['prompts']}")
            save_population(iter+1, sorted_population, root_folder, list(range(n_pop)))
            best_score_iterations.append(max(sorted_population['eval']))
        # increase iter counter
        iter += 1

    # Create a new dictionary with the same keys, but values are lists with only the selected indices
    best_pop, keep_list = pop_selection(sorted_population, 1, 1)
    if save == True:
            save_population('best', best_pop, root_folder, [0])
            print(f"Data saved for iteration best.")
            end_time = datetime.now()
            save_details_alg_2(root_folder, n_pop, 
                            n_top, # no. of top elements being kept
                            start_time, 
                            end_time,
                            patience,
                            max_iter,
                            iter,
                            best_score_iterations,
                            eval_data,
                            data_size,
                            task,
                            model_name,
                            quantize_model_4bits,
                            alg=task,
                            N=N,
                            eval_mutation_prob=eval_mutation_prob,
                            evaluation_task=evaluation_task,)
            
            create_plots_from_RUNS_folder(root_folder)

    return best_pop, best_score_iterations
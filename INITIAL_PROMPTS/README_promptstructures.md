# SemEval

## Basic setup

- task descriprion
- ctr description
- PRIMARY CTR's SECTION (from dataset)
- (SECONDARY CTR's SECTION) (from dataset)
- statement description
- STATEMENT (from dataset)
- answer description

## Setup with retrieved highlights

- task descriprion
- ctr description
- PRIMARY CTR's SECTION (from dataset)
- (SECONDARY CTR's SECTION) (from dataset)
- statement description
- STATEMENT (from dataset)
- highlights description
- PRIMARY CTR HIGHLIGHTS (retrieved from the CTR's section)
- SECONDARY CTR HIGHLIGHTS (retrieved from the CTR's section)
- answer description

## Setup with self-reasoning

### To get the REASONING CHAIN

- task descriprion
- ctr description
- PRIMARY CTR's SECTION (from dataset)
- (SECONDARY CTR's SECTION) (from dataset)
- statement description
- STATEMENT (from dataset)
- self reasoning prompt A (asks for the resoning chain)

### Prompt for the answer using the REASONING CHAIN

- task descriprion
- ctr description
- PRIMARY CTR's SECTION (from dataset)
- (SECONDARY CTR's SECTION) (from dataset)
- statement description
- STATEMENT (from dataset)
- self reasoning prompt B (description of the resoning chain)
- REASONING CHAIN
- self reasoning prompt C (prompts for the answer while asking to take the reasoning chain into account)

# Contract NLI

## Setup with only oracle spans

- task descriprion
- doc description (describes the NDA's sections)
- ORACLE NDA SPANS (from dataset)
- statement description
- STATEMENT (from dataset)
- answer description

## Setup with full NDA and oracle spans or retrived spans

- task descriprion
- full doc description (not added yet)
- FULL NDA (from dataset)
- ORACLE NDA SPANS or RETRIEVED NDA SPANS(from dataset)
- statement description
- STATEMENT (from dataset)
- answer description

# MEDIQA SUM

## Basic Setup

- task descriprion
- dialogue description
- DIALOGUE (from dataset)
- example description
- RETRIEVED CLINICAL NOTE (from dataset)
- answer description

# CSQA (Not relevant for now)

- task descriprion
- QUESTION (from dataset)
- answer description

# Evolutionary Prompts

OLD VERSION in combination_prompts.txt and muatation_prompts.txt - had only one part instead of the 3

NEW VERSION (with 3 parts)
- task descriprion
- instruction description
- INSTRUCTION/INSTRUCTIONS
- answer description

NEW VERSION in the folders INITIAL_PROMPTS/evolutionary_prompts/mutation and INITIAL_PROMPTS/evolutionary_prompts/combination

It contains the prompts that perform the mutations and the crossovers of the promtps, in the basic setup only the first ones from each are used. The other 4 are used as a initial population for the hyperevolution of the mutation and crossover prompts. 

Each subprompt is trying to explain the task of performing the mutation/crossover operation in different ways. Ranging from giving the LLM more freedom (exploratory) to asking for only a slight rewriting of the original instruction (exploit). Below are what each individual description is trying to acomplish. ie: the 2nd description in the task_description, instruction_description and answer_description is trying to ask the LLM for a more condensed version of the original instruction.

## Mutation Prompts

1. Base
2. Shorter Prompt
3. Exploratory
4. Conservative
5. Adds positive reinforcement

## Crossover Prompts

1. Base
2. Combine the best parts while being concise
3. Keep everything from both prompts
4. Add positive reinforcement
5. Take ideas from both and build onto them freely

# Hyper Evolutionary Prompts

DEPRECATED - was being use to perform the mutation and crossover operations on the mutation/crossover prompts during the hyper mutation
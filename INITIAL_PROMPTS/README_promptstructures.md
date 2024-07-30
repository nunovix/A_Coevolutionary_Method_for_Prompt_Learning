# SemEval

## Basic setup

- task descriprion
- ctr description
- PRIMARY CTR's SECTION [from dataset]
- (SECONDARY CTR's SECTION) [from dataset]
- statement description
- STATEMENT [from dataset]
- answer description

## Setup with retrieved highlights

- task descriprion
- ctr description
- PRIMARY CTR's SECTION [from dataset]
- (SECONDARY CTR's SECTION) [from dataset]
- statement description
- STATEMENT [from dataset]
- highlights description
- PRIMARY CTR HIGHLIGHTS [retrieved from the CTR's section]
- SECONDARY CTR HIGHLIGHTS [retrieved from the CTR's section]
- answer description

## Setup with self-reasoning

### To get the REASONING CHAIN

- task descriprion
- ctr description
- PRIMARY CTR's SECTION [from dataset]
- (SECONDARY CTR's SECTION) [from dataset]
- statement description
- STATEMENT [from dataset]
- self reasoning prompt A - asks for the resoning chain

### Prompt for the answer using the REASONING CHAIN

- task descriprion
- ctr description
- PRIMARY CTR's SECTION [from dataset]
- (SECONDARY CTR's SECTION) [from dataset]
- statement description
- STATEMENT [from dataset]
- self reasoning prompt B - description of the resoning chain
- REASONING CHAIN
- self reasoning prompt C - asks for the answer while asking to take the reasoning chain into account

# Contract NLI

## Setup with full NDA + oracle spans (2 labels)

- task descriprion
- doc description
- FULL NDA [from dataset]
- highlight description
- ORACLE NDA SPANS [from dataset]
- statement description
- STATEMENT [from dataset]
- answer description (2 labels)

## Setup with full NDA + retrieved spans (2 labels)

- task descriprion
- doc description
- FULL NDA [from dataset]
- RETRIEVED NDA SPANS (using the model from hf Alibaba-NLP/gte-large-en-v1.5)
- statement description
- STATEMENT [from dataset]
- answer description (2 labels)

## Setup with full NDA + retrieved spans (3 labels)

- task descriprion
- doc description
- FULL NDA [from dataset]
- 4 RETRIEVED NDA SPANS (using the model from hf Alibaba-NLP/gte-large-en-v1.5)
- statement description
- STATEMENT [from dataset]
- answer description (3 labels)

# MEDIQA SUM

## Basic Setup

- task descriprion
- dialogue description
- DIALOGUE [from dataset]
- example description
- RETRIEVED CLINICAL NOTE (using the model from hf Alibaba-NLP/gte-large-en-v1.5 on the dialogues)
- answer description

# CSQA (Not relevant for now)

- task descriprion
- QUESTION [from dataset]
- answer description

# Evolutionary Prompts

OLD VERSION in combination_prompts.txt and muatation_prompts.txt - had only one part instead of the 3 (kept for now just in case)

NEW VERSION in the folders INITIAL_PROMPTS/evolutionary_prompts/mutation and INITIAL_PROMPTS/evolutionary_prompts/combination

NEW VERSION (with 3 parts)
- task descriprion
- instruction description
- INSTRUCTION/INSTRUCTIONS
- answer description

It contains the prompts that perform the mutations and the crossovers of the promtps, in the basic setup only the first ones from each are used. The other 4 are used as a initial population for the hyperevolution of the mutation and crossover prompts. 

Each subprompt is trying to explain the task of performing the mutation/crossover operation in different ways. Ranging from giving the LLM more freedom (exploratory) to asking for only a slight rewriting of the original instruction (exploit). Below are what each individual description is trying to acomplish. ie: the 2nd description in the task_description, instruction_description and answer_description is trying to ask the LLM for a more condensed version of the original instruction.

## Mutation Prompts

1. Base
2. Shorter Prompt
3. Asks for words to be replaced by synonyms
4. Asks for the order of the sentences to be changed
5. Adds positive reinforcement

## Crossover Prompts

1. Base
2. Combine the best parts while being concise
3. Keep everything from both prompts
4. Add positive reinforcement
5. Take ideas from both and build onto them freely

# Hyper Evolutionary Prompts

NOT USED - was being use to perform the mutation and crossover operations on the mutation/crossover prompts during the hyper mutation
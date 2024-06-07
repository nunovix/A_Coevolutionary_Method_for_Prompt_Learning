# SemEval

## Basic setup

-task descriprion
-ctr description
-PRIMARY CTR's SECTION (from dataset)
-(SECONDARY CTR's SECTION) (from dataset)
-statement description
-STATEMENT (from dataset)
-answer description

## Setup with retrieved highlights

-task descriprion
-ctr description
-PRIMARY CTR's SECTION (from dataset)
-(SECONDARY CTR's SECTION) (from dataset)
-statement description
-STATEMENT (from dataset)
-highlights description
-PRIMARY CTR HIGHLIGHTS (retrieved from the CTR's section)
-SECONDARY CTR HIGHLIGHTS (retrieved from the CTR's section)
-answer description

## Setup with self-reasoning

### To get the REASONING CHAIN

-task descriprion
-ctr description
-PRIMARY CTR's SECTION (from dataset)
-(SECONDARY CTR's SECTION) (from dataset)
-statement description
-STATEMENT (from dataset)
-self reasoning prompt A (asks for the resoning chain)

### Prompt for the answer using the REASONING CHAIN

-task descriprion
-ctr description
-PRIMARY CTR's SECTION (from dataset)
-(SECONDARY CTR's SECTION) (from dataset)
-statement description
-STATEMENT (from dataset)
-self reasoning prompt B (description of the resoning chain)
-REASONING CHAIN
-self reasoning prompt C (prompts for the answer while asking to take the reasoning chain into account)

# Contract NLI

## Setup with only oracle spans

-task descriprion
-doc description (describes the NDA's sections)
-ORACLE NDA SPANS (from dataset)
-statement description
-STATEMENT (from dataset)
-answer description

## Setup with full NDA and oracle spans or retrived spans

-task descriprion
-full doc description (not added yet)
-FULL NDA (from dataset)
-ORACLE NDA SPANS or RETRIEVED NDA SPANS(from dataset)
-statement description
-STATEMENT (from dataset)
-answer description

# MEDIQA SUM

## Basic Setup

-task descriprion
-dialogue description
-DIALOGUE (from dataset)
-example description
-RETRIEVED CLINICAL NOTE (from dataset)
-answer description

# CSQA

Not relevant for now

# Evolutionary Prompts

Contains the prompts that perform the mutations and the crossovers of the promtps, in the basic setup only the first ones from each are used. The other 4 are used as a initial population for the hyperevolution of the mutation and crossover prompts.

# Hyper Evolutionary Prompts

For now they perform the mutation and crossover of the mutation and crossover prompts in the hyper evolutionary setting.
#extract_LEXSUM_data(used_retrieved_file = False)

import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

from evo_functions import convert_text_mistral_phi3

torch.random.manual_seed(0) 
model = AutoModelForCausalLM.from_pretrained( 
    "microsoft/Phi-3-mini-4k-instruct",  
    device_map="cuda",  
    torch_dtype="auto",  
    trust_remote_code=True,  
    attn_implementation="flash_attention_2"
) 

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct") 

sentence = """[INST]Consider the task of writing a summary of a civil lawsuit from the American legal system. A civil lawsuit involves a set of actions among two or more parties and the judge(s). The summary should have only one paragraph with a brief description of the background, parties envolved, and the outcome (so far) of the case.

Review the following example of a summary for a different civil lawsuit. The summary you will be generating should have a similar structure and length.

Example Summary:
"Clean and energy-efficient road transport vehicles
Clean and energy-efficient road transport vehicles
SUMMARY OF:
Directive 2009/33/EC on promoting clean and energy-efficient road transport vehicles
WHAT IS THE AIM OF THE DIRECTIVE?
It aims at promoting and stimulating the development of a market for clean and energy-efficient vehicles in the European Union (EU).
As amended by Directive (EU) 2019/1161, it sets minimum national targets for the procurement of clean vehicles. The targets are defined as a minimum share of clean vehicles in the aggregate public authorities and certain public transport operators across a Member State.
KEY POINTS
Scope
As amended by Directive (EU) 2019/1161, the directive applies to contracts for the procurement of certain road transport vehicles (cars, vans, trucks and buses) and services, by contracting authorities, contracting entities and operators of public service obligations under a public service contract.
The amending directive extended the application of the directive so that it now covers procurement through:
contracts for the purchase, lease, rent and hire-purchase of vehicles;
service contracts for public road passenger transport;
service contracts for public road transport services, special-purpose road passenger-transport services, non-scheduled passenger transport, refuse collection services, and mail and parcel transport and delivery.
Amending Directive (EU) 2019/1161 also introduced a new definition of a clean vehicle*.
Minimum procurement targets
Directive 2009/33/EC, as amended by Directive (EU) 2019/1161, sets minimum public procurement targets for light-duty vehicles (cars and vans), trucks and buses for 2025 and 2030. In the case of buses, half of the targets have to be met with zero-emission vehicles (battery electric or hydrogen buses). These targets are set out in the directives annex.
For each Member State, a different target is set for light-duty vehicles, trucks and buses. These targets are calculated as a minimum percentage of clean vehicles in the total number of vehicles procured through public procurement in each Member State, over two 5-year periods: 20212025 and 20262030. Member States have to ensure that the targets are met, but they have full flexibility in how they distribute the effort across different contracting authorities and contracting entities.
Sharing best practice
The European Commission encourages the sharing of knowledge and best practice between Member States with a view to promoting the purchase of clean and energy-efficient road transport vehicles.
Several initiatives are under way that ensure the directive is implemented. These include:
guidelines on green public procurement and a technical background report;
the European green vehicle initiative, which seeks to support the development of green vehicles and sustainable mobility solutions;
the European clean bus deployment initiative; and
various studies.
Committee procedure
The Commission has the power to adopt implementing acts and is assisted in this by a committee governed by the EUs comitology rules.
FROM WHEN DOES THE DIRECTIVE APPLY?
              
The directive has applied since 4 June 2009 and had to become law in the Member States by 4 December 2010.
The changes introduced by amending Directive (EU) 2019/1161, including the introduction of a clean vehicle definition and the setting of minimum national targets for their procurement, have applied since 2 August 2019, and had to become law in the Member States by 2 August 2021.
BACKGROUND
              
For further information, see:
Clean vehicles directive (European Commission).
KEY TERMS
              
Clean vehicle.A clean light-duty vehicle (e.g. car, van) is defined on the basis of its CO2 emissions (the applicable emission limits are laid down in Table 2 of the directives annex).A clean heavy-duty vehicle (e.g. bus, truck) is defined on the basis of the use of alternative fuels, as defined in Directive 2014/94/EU (see summary). A separate definition is provided for zero-emission heavy-duty vehicles.
MAIN DOCUMENT
            
Directive 2009/33/EC of the European Parliament and of the Council of 23 April 2009 on the promotion of clean and energy-efficient road transport vehicles (OJ L 120, 15.5.2009, pp. 512).
Successive amendments to Directive 2009/33/EC have been incorporated in the original text. This consolidated version is of documentary value only.
RELATED DOCUMENTS
            
Directive 2014/94/EU of the European Parliament and of the Council of 22 October 2014 on the deployment of alternative fuels infrastructure (OJ L 307, 28.10.2014, pp. 120).
See consolidated version.
Report from the Commission to the European Parliament, the Council, the European Economic and Social Committee and the Committee of the Regions on the application of Directive 2009/33/EC on the promotion of clean and energy efficient road transport vehicles (COM (2013) 214 final, 18.4.2013).
Commission Notice on the application of Articles 2, 3, 4 and 5 of Directive 2009/33/EC of the European Parliament and of the Council on the promotion of clean road transport vehicles in support of low-emission mobility 2020/C 352/01 (OJ C 352, 22.10.2020, pp. 112).
last update 10.02.2022" 

The civil lawsuit, which can be comprised of several documents, is given next.

"25.7.2019
EN
Official Journal of the European Union
L 198/202
REGULATION (EU) 2019/1242 OF THE EUROP"

Now, generate the summary of the civil lawsuit without any additional explanations.[/INST]

Summary:
"""

sentence = convert_text_mistral_phi3(sentence)

encoded_inputs = tokenizer(sentence,return_tensors="pt", return_attention_mask=True, padding=True).to('cuda')
input_len = encoded_inputs['input_ids'][0].shape[0]
output = model.generate(encoded_inputs['input_ids'], attention_mask=encoded_inputs['attention_mask'], max_new_tokens=500)

t = tokenizer.decode(output[0], skip_special_tokens=False)

print(f"output--->{output}")
print(f"t--->{t}")



"""
messages = [ 
    {"role": "system", "content": "You are a helpful AI assistant."}, 
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
] 

pipe = pipeline( 
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
) 

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": True, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])
"""
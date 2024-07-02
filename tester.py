import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trie import MarisaTrie

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

class MyMarisaTrie(MarisaTrie):
    def __init__(self, data): super().__init__(data)
    def get(self, data, length_to_ignore): return super().get([tokenizer.bos_token_id] + data[length_to_ignore:])

trie = MyMarisaTrie([[tokenizer.bos_token_id] + tokenizer.encode("Yes.") + [tokenizer.eos_token_id],
                     [tokenizer.bos_token_id] + tokenizer.encode("Maybe.") + [tokenizer.eos_token_id],
                     [tokenizer.bos_token_id] + tokenizer.encode("No.") + [tokenizer.eos_token_id]])

prompt = tokenizer.encode("<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nQuestion?<|end|>\n<|assistant|>\n ", return_tensors="pt")
prompt_length = prompt[0].shape[0]

print("Constrained output...")
output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_length=20, prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist(), prompt_length))
print(tokenizer.decode(output[0]))

print("Unconstrained output...")
output = model.generate(prompt, pad_token_id=tokenizer.eos_token_id, max_length=20)
print(tokenizer.decode(output[0]))

print("Using cached results...")
cached_outputs = model(prompt, return_dict=True,)
cached_outputs.past_key_values = [[y[:, :, :-1] for y in x] for x in cached_outputs.past_key_values]
output = model.generate(prompt, past_key_values=cached_outputs.past_key_values, pad_token_id=tokenizer.eos_token_id, max_new_tokens=15, use_cache=True,)
print(tokenizer.decode(output[0]))
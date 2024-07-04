import os
# set available gpu's
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from evo_functions import load_model

model, tokenizer = load_model(checkpoint = "microsoft/Phi-3-mini-4k-instruct", quantized = True)


bos = [tokenizer.bos_token_id]
eos = [tokenizer.eos_token_id]

print(f"BOS-->{bos}")
print(f"EOS-->{eos}")

print("after decoding")
bos_conv = tokenizer.decode(bos, skip_special_tokens=False)
eos_conv = tokenizer.decode(eos, skip_special_tokens=False)
print(f"BOS after-->{bos_conv}")
print(f"EOS after-->{eos_conv}")
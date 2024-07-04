import numpy as np

# for sampling with softmax and a given sampling Temperature, if none is provided a equal probability will be given to all elements
def softmax_samp_T(x, sampling_T = 5.0):

    if sampling_T == None or sampling_T == 0:
        return [1/len(x)] * len(x)

    x = np.array(x)
    # if values in decimal form convert to percentage so that sampling T works as desired
    if max(x) < 1:
        x = 100 * x

    # apply sampling T
    x = x/sampling_T

    return np.exp(x) / np.sum(np.exp(x), axis=0)

a = [0.72, 0.71, 0.70, 0.66, 0.59]

print(a)
print(softmax_samp_T(a, 5))
print(softmax_samp_T(a, 10))
print(softmax_samp_T(a, 20))


"""import os
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
print(f"EOS after-->{eos_conv}")"""
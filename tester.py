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


a = [0.72, 0.58, 0.5, 0.68, 0.65]
b = [72, 58, 50, 68, 65]
print(f"a-->{a}")
print(f"softmax_samp_T(a, 1.0)-->{softmax_samp_T(a, 1.0)}")
print(f"a-->{a}")
print(f"softmax_samp_T(a, 5.0)-->{softmax_samp_T(a, 5.0)}")
print(f"a-->{a}")
print(f"softmax_samp_T(a, 10.0)-->{softmax_samp_T(a, 10.0)}")
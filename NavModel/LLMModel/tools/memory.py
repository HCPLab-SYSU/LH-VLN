import numpy as np
import math
import random


def forget(step, history, hist_vis, random_forget):
    if random_forget:
        first_five_indices = list(range(5))
        random_index = random.choice(first_five_indices)
        hist_vis[0].pop(random_index)
        history[0].pop(random_index)
        history[1].pop(random_index)
    else:
        weight = []
        for i in range(len(hist_vis[0])):
            max_probability = np.amax(history[0][i])
            min_probability = np.amin(history[0][i])
            normalized_prob_vector = (history[0][i] - min_probability) / (max_probability - min_probability)
            confidence = np.amax(normalized_prob_vector)

            time_prob = 1 - (step - history[1][i])/step
            time_exp = -math.exp(-time_prob) + 1 + 1 / math.e

            weight.append(confidence * time_exp)
        
        min_index = min(enumerate(weight), key=lambda x: x[1])[0]
        hist_vis[0].pop(min_index)
        history[0].pop(min_index)
        history[1].pop(min_index)
    
    return history, hist_vis

import torch
def avg_pooling(v, index):
    if index == 0:
        v = [(v[0] + v[1])/2.0] + v[2:]
    elif index == len(v) - 1:
        v = v[:-2] + [(v[-2] + v[-1])/2.0]
    else:
        v = v[:index - 1] + [(v[index-1] + v[index])/2.0, (v[index] + v[index + 1])/2.0] + v[index+2:]
    return v
    
def calculate_entropy(vs):
    v = [t.view(-1) for t in vs]
    v = torch.cat(v)

    v_normalized = v / v.sum()
    entropy = -torch.sum(v_normalized * torch.log2(v_normalized + 1e-10))

    return entropy

def forget_with_entropy(history, hist_vis_):
    hist_prob = history
    hist_vis = hist_vis_
    key_index = 0
    key_entropy = math.inf
    for i in range(len(hist_prob)):
        entropy = calculate_entropy(avg_pooling(hist_prob, i))
        if entropy < key_entropy:
            key_entropy = entropy
            key_index = i

    history = avg_pooling(hist_prob, key_index)
    hist_vis_ = avg_pooling(hist_vis, key_index)

    return history, hist_vis_


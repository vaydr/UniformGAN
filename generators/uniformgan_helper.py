from typing import Counter
import numpy as np
import math
import torch
import torch.nn as nn

def compute_privacy_sigma(epsilon, delta, mM):
    return (2*(mM)*math.sqrt(math.log(1/delta)))/epsilon

def calculate_epsilon_helpers(epsilon):
    if (epsilon is None):
        epsilon = 0.0003
    return epsilon, 1 - epsilon, 1 - 2*epsilon

def compute_transform_column_int(data, epsilon, special_order=None):
    epsilon, epsilon_high, epsilon_modifier = calculate_epsilon_helpers(epsilon)
    # Should be 0, EPSILON, normal range, 1-EPSILON, 1
    dist, _ = np.histogram(data, bins=sorted(list(set(data)) + [float('inf')]))
    keys = sorted(list(set(data)))
    if (special_order): # Oridinal discrete columns
        keys = special_order
    new_dist = [epsilon]
    total = sum(dist)
    acruid = 0
    for i in range(len(dist)):
        p = dist[i]/total * epsilon_modifier
        acruid += p
        new_dist.append(acruid)
    new_dist[-1] = epsilon_high
    def lam(number):
        for i in range(1, len(keys)+1):
            if (number >= new_dist[i]):
                continue
            return i-1
        return len(keys) - 1
    return {'dist': new_dist, 'keys': keys, 'lambda': lam, 'type': 'categorical' }

def compute_transform_column_str(data, epsilon, special_order=None):
    epsilon, epsilon_high, epsilon_modifier = calculate_epsilon_helpers(epsilon)
    dist = dict(data.value_counts())
    if (special_order): # Oridinal discrete columns
        keys = special_order
    else:
        keys = list(dist)
    new_dist = [epsilon]
    total = len(data)
    acruid = 0
    for i in range(len(keys)):
        p = dist[keys[i]]/total * epsilon_modifier
        acruid += p
        new_dist.append(acruid)
    new_dist[-1] = epsilon_high
    def lam(number):
        for i in range(1, len(keys)+1):
            if (number >= new_dist[i]):
                continue
            return i-1
        return len(keys) - 1
    return {'dist': new_dist, 'keys': keys , 'lambda': lam, 'type': 'categorical' }


def compute_transform_column_continuous(data, epsilon, quantile_count=40, delta=1e-8):
    epsilon, _, epsilon_modifier = calculate_epsilon_helpers(epsilon)
    # Every 0.025 we'll count
    pure = [i/quantile_count for i in range(quantile_count+1)] #Includes 0 and 1
    quantiles = np.quantile(data, pure)
    i = 1
    buckets = {}
    while i <= quantile_count + 1:
        count = 1
        while(i + count <= quantile_count + 1 and quantiles[i-1] == quantiles[i+count-1]):
            count += 1
        if (count > 1):
            buckets[i] = count - 1
        i += count
    # Add two buffer buckets
    pure = [0] + list(map(lambda x: (x*epsilon_modifier) + epsilon, pure)) + [1]
    quantiles = [(quantiles[0]*2 - quantiles[1])] + list(quantiles) + [(2*quantiles[-1] - quantiles[-2])]
    res = {'quantile_count': quantile_count, 'quantile_dx': 1/quantile_count*epsilon_modifier,
        'placements': quantiles, 'buckets': buckets, 'delta': delta,
        'type': 'continuous' }
    return res


def compute_create_mapping(compute):
    mapping = {}
    keys = compute['keys']
    dist = compute['dist']
    for i in range(len(keys)):
        def key_helper(i):
            r = (dist[i + 1] - dist[i]) / 2
            return lambda: (np.random.uniform(0, 1, 1)[0] * r * 2) + dist[i]
        mapping[keys[i]] = key_helper(i)
    def helper(number):
        val = mapping[number]()
        return val
    return helper


def transform_column_int(data, compute=None, special_order=None, epsilon=None):
    """
    Outputs a tuple of the transformed column and compute dictionary
    """

    if (compute is None):
        compute = compute_transform_column_int(data, epsilon, special_order=special_order)
    return data.map(compute_create_mapping(compute)), compute

def transform_column_str(data, compute=None, special_order=None, epsilon=None):
    """
    Outputs a tuple of the transformed column and compute dictionary
    """
    if (compute is None):
        compute = compute_transform_column_str(data, epsilon, special_order=special_order)
    return data.map(compute_create_mapping(compute)), compute


def transform_column_continuous(data, compute=None, epsilon=None):
    if (compute is None):
        compute = compute_transform_column_continuous(data,epsilon)
    epsilon, _, _ = calculate_epsilon_helpers(epsilon)
    dx = compute['quantile_dx']
    delta = compute['delta']
    placements = compute['placements']
    buckets = compute['buckets']
    def helper(number):
        # Can never be i == 0 because of bounds!
        for i, placement in enumerate(placements):
            # Practically the same
            if abs(number - placement) < delta and i in buckets:
                return (np.random.uniform(0, 1, 1)[0] * dx * buckets[i]) + (i-1) * dx + epsilon
            elif number > placement:
                # do nothing
                pass
            elif number < placement:
                # Perfect bucket
                if i == 0: # min bound
                    return 0
                mini = i - 1
                minplacement = placements[mini]
                range_ = placement - minplacement

                # do the linear interpolation of the value
                return (number - minplacement)/range_ * dx + (mini - 1)*dx + epsilon 
        return 0
    return data.map(helper), compute


def reverse_transform_column(transformed, compute, epsilon=None):
    epsilon, _, _ = calculate_epsilon_helpers(epsilon)
    type_ = compute['type']
    if (type_ == "continuous"):
        dx = compute['quantile_dx']
        delta = compute['delta']
        placements = compute['placements']
        def key_helper(number):
            def calcI(i):
                if (i <= 1):
                    return i * epsilon 
                else:
                    return (i - 1)*dx + epsilon
            # dx doesn't compute at the beginning
            for i, placement in enumerate(placements):
                # Practically the same
                if abs(number - calcI(i)) < delta:
                    return placement
                elif number > calcI(i):
                    # do nothing
                    continue
                # Perfect bucket
                if i == 0: # min bound
                    return placement
                mini = i - 1
                minplacement = placements[mini]
                range_ = placement - minplacement
                # do the linear interpolation of the value
                ddx = dx
                if mini == 0:
                    ddx = epsilon 
                return (number - calcI(mini)) / ddx * range_ + minplacement
            return 1
        return transformed.map(key_helper)

    # CATEGORICAL
    keys = compute['keys']
    lam = compute['lambda']
    def helper(number):
        return keys[lam(number)]
    return transformed.map(helper)

def compute_sigma(privacy_epsilon, privacy_delta, privacy_mM):
    return (2*(privacy_mM)*math.sqrt(math.log(1/privacy_delta)))/privacy_epsilon

def transform_dataset(js, dataset):
    normalizer = {}
    newdataset = dataset.copy()
    columns = list(dataset.keys())
    discrete_columns = js['discrete_columns']
    ordinal_discrete_columns = js['ordinal_discrete_columns']
    additionals = js['additionals']
    epsilon = additionals("epsilon") or None
    for i in range(len(columns)):
        uniques = list(dataset[columns[i]].unique())
        column_name = columns[i]
        compute = None
        if (column_name in discrete_columns or any((isinstance(x, str)) for x in uniques)): # Saving yourself if you accidently say a column is discrete
            special_order = None
            if (column_name in ordinal_discrete_columns):
                special_order = ordinal_discrete_columns[column_name]
            if (all((not isinstance(x, str)) for x in uniques)):
                newdataset[columns[i]], compute = transform_column_int(dataset[columns[i]], special_order=special_order, epsilon=epsilon)
            else:
                newdataset[columns[i]], compute = transform_column_str(dataset[columns[i]], special_order=special_order, epsilon=epsilon)
        else:
            newdataset[columns[i]], compute = transform_column_continuous(dataset[columns[i]], epsilon=epsilon)
        newdataset[columns[i]] = newdataset[columns[i]] * 2 - 1
        normalizer[i] = {
            'name': columns[i],
            'max': dataset[columns[i]].max(),
            'min': dataset[columns[i]].min(),
            'uniques': len(uniques),
            'compute': compute,
        }
    return normalizer, newdataset


def convolution(inp, x):
    (kernel, stride, padding) = x
    return (inp - kernel + 2*padding)/stride + 1

def convolution_transpose(inp, x):
    (kernel, stride, padding, output_padding) = x
    return (inp - 1)*stride - 2*padding + (kernel - 1) + (output_padding + 1)
            # 1 - 1 - 2 * 0 + 3 

class Convolutions():
    div3 = (4, 2, 0)
    min4 = (5, 1, 0)
    odddiv2 = (3, 2, 1) # (x + 1) / 2
    div2 = (4, 2, 1)
    same = (3,1,1)
    tmult2 = (3, 2, 1, 1) # Double
    toddmult2 = (3, 2, 1, 0) # Double
    minx = lambda x: (x+1, 1, 0)


def get_convolutions(inp):
    kdiscrim = []
    kdiscrim.append(Convolutions.same)
    kgenerator = []
    while (inp > 1):
        if (inp % 2 == 1): # ODD
            kdiscrim.append(Convolutions.odddiv2)
            kgenerator.append(Convolutions.toddmult2)
        else:
            kdiscrim.append(Convolutions.div2)
            kgenerator.append(Convolutions.tmult2)
        inp = convolution(inp, kdiscrim[-1])
    kgenerator.reverse()
    return kgenerator, kdiscrim


def mul(x):
    x = list(x)
    s = 1
    for i in x:
        s *= i
    return s

def get_mask(module, sparsity_val=0.3):
    weights = module.weight
    count = mul(weights.shape)
    original_shape = weights.shape
    flatten = weights.flatten().abs()
    topk = torch.topk(flatten, int(count * sparsity_val), largest=False)
    mask = torch.ones_like(flatten)
    mask[topk.indices] = 0
    mask = mask.reshape(original_shape)
    return mask
    
def get_masks_from_sequence(seq):
    masks = []
    for module in seq:
        if (type(module) == nn.Conv2d):
            masks.append(get_mask(module))
    return masks

def apply_masks_from_sequence(seq, masks):
    i = 0
    for module in seq:
        if (type(module) == nn.Conv2d):
            d = module.state_dict()
            d['weight'] = d['weight'] * masks[i]
            module.load_state_dict(d)
            i += 1
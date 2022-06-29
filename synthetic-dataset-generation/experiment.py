import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random

def internal_consistency(d):
	'''
	IMPORTANT: THIS IMPLEMENTATION OF INTERNAL CONSISTENCY IS LINEAR IN THE SUM OF ARRAY SIZES

	If you have an array A with size M, and B with size N, this implementation of consistency is O(N+M).

	d: pandas dataframe. returns an integer.
	'''

	if not d:
		return 0


	if type(d[0]) == type([]):
		return internal_consistency([j for i in d for j in i])

	hash =[{"0":0, "1":0} for i in range(32)]
	for ele in d:
		bits = bin(ele)[2:]			
		bits = "0" * (32 - len(bits)) + bits
		for i in range(32):
			hash[i][bits[i]] +=1
	
	dist = 0
	for i in range(32):
		dist += hash[i]["0"] * hash[i]["1"]  
	
	return dist

def generative_power(a,b):
	'''
	ASSUME that data is random, and crucially, UNSORTED, so that line of best fit has zero slope (on index vs. generative power)
	'''
	return abs(internal_consistency(a)-internal_consistency(b))

def kl_divergence(dataset, sample_dataset):
    from scipy.stats import ks_2samp, entropy
    columns = list(dataset.keys())
    print("[INFO]\tKL Divergence\n")
    kldivergence = []
    for i in range(len(columns)):
        column = columns[i]
        val = None
        try:
            val = 1 - ks_2samp(dataset[column], sample_dataset[column]).statistic
        except:
            try:
                val = 1 - ks_2samp(dataset[column].astype(str), sample_dataset[column].astype(str)).statistic
            except:
                print(f"[ERROR]\tCould not calculate column\t{column}\n")
                traceback.print_exc(file=error_logger)
        if val is not None:
            ksdivergence.append(val)
            print(f"[SUBINFO]\tKS\t{column}\t{val}\n")

        kl = None
        try:
            minlength = min(len(dataset[column]), len(sample_dataset[column]))
            kl = entropy(dataset[column][:minlength], sample_dataset[column][:minlength])
        except TypeError:
            pass # non float lines
        except:
            traceback.print_exc()
            pass
        if (kl is not None): 
            print(f"[SUBINFO]\tKL\t{column}\t{kl}\n")
            if (kl != float('inf')):
                kldivergence.append(kl)
    print(f"[STAT]\tAverage KS-Divergence:\t{sum(ksdivergence)/len(ksdivergence)}\n")
    if len(kldivergence) > 0:
        print(f"[STAT]\tAverage Non-infinity KL-Divergence:\t{sum(kldivergence)/len(kldivergence)}\n")
        return sum(kldivergence)/len(kldivergence)
    else:
        print(f"[STAT]\tAverage Non-infinity KL-Divergence:\tN/A\n")
        return 

if __name__ == "__main__":
	PROP = 0.75 + 0.1*random.random()
	df = pd.read_csv('upload/adult.csv')
	data = df[["age", "fnlwgt"]]
	x = data.iloc[0:1000]

	x_ax = []
	y_ax = []

	for i in range(500):
		n = random.randint(1000,30000)
		y = data.iloc[n:n+1000]

		x_ax.append(n)
		y_ax.append(generative_power(x.values.tolist(), y.values.tolist()))
	z_ax = [(i + random.random()*i**1.4 - i/2) for i in y_ax]
	z = [i/(PROP*max(z_ax)) for i in z_ax]

	plt.scatter(y_ax, z)
	plt.show()




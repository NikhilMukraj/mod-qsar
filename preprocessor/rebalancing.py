import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from smiles_tools import SmilesEnumerator
import sys
import argparse


GREEN = '\033[1;32m'
NC = '\033[0m'
RED = '\033[0;31m'

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', help='Name of .csv file to parse', required=True)
parser.add_argument('-o', '--output', help='Output name of rebalanced data', required=True)
parser.add_argument('-n', '--num_bins', help='Number of bins to use in rebalancing', type=int, default=20)
parser.add_argument('-s', '--standardize', help='Standardize data', type=bool, default=False)

args = parser.parse_args()

if args.num_bins < 2:
    print(f"{RED}`num_bins` argument must be an integer least 2 or greater{NC}")
    sys.exit(1)

df = pd.read_csv(args.input)

if 'ACTIVITY' not in df.columns:
    print(f"{RED}'ACTIVITY' column required{NC}")
    sys.exit(1)
if 'SMILES' not in df.columns:
    print(f"{RED}'SMILES' column required{NC}")
    sys.exit(1)

if not is_numeric_dtype(df['ACTIVITY']):
    print(f"{RED}'ACTIVITY' column must have all numeric regression values{NC}")
    sys.exit(1)

binding_tuples = [(n, i[1][1]) for n, i in enumerate(df.iterrows())]

unprocessed_counts = {}
for i in binding_tuples:
    if i[0] not in unprocessed_counts.keys():
        unprocessed_counts[i[0]] = [i[1]]
    else:
        unprocessed_counts[i[0]].append(i[1])

counts = {}
for key, value in unprocessed_counts.items():
    counts[key] = min(value)

no_dups_count = [(df['SMILES'][key], value[0]) for key, value in unprocessed_counts.items() if len(value) == 1]

hist = np.histogram([i[1] for i in no_dups_count], bins=args.num_bins)

def augment_smiles(string, n):
    sme = SmilesEnumerator()
    output = []
    for i in range(n):
        output.append(sme.randomize_smiles(string))
    
    return output

get_bucket = lambda n, tuples : [i for i in tuples if i[1] > hist[1][n] and i[1] < hist[1][n+1]]

bucket_range = len(hist[0]) - 1
all_strings = []
for i in range(bucket_range):
    bucket = get_bucket(i, no_dups_count)
    strings = []
    for j in bucket:
        strings += list(set([(string, j[1]) for string in augment_smiles(j[0], int(max(hist[0]) / len(bucket)))]))
        
    all_strings += strings

output_df = pd.DataFrame(all_strings, columns=['SMILES', 'ACTIVITY'])
if args.standardize:
    scaler = (1 - 0) / (df['ACTIVITY'].max() - df['ACTIVITY'].min())
    output_df['ACTIVITY'] = (output_df['ACTIVITY'] - df['ACTIVITY'].min()) * scaler

output_df.to_csv(args.output, index=None)

print(f"{GREEN}Finished rebalancing input data{NC}")

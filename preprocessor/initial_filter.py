import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from SmilesPE.pretokenizer import atomwise_tokenizer


if len(sys.argv) < 2:
    print('Too few args')
    sys.exit()

df = pd.read_csv(f'{sys.argv[1]}', low_memory=False)

df = df[['SMILES', 'ACTIVITY']]

df = df.loc[df['ACTIVITY'] != 'Inconclusive', :]
df = df[~df['SMILES'].isnull()]

activeDF = df.loc[df['ACTIVITY'] == 'Active', :]
inactiveDF = df.loc[df['ACTIVITY'] == 'Inactive', :]

if len(activeDF) <= len(inactiveDF):
    filtered = pd.concat([activeDF.loc[:, 'SMILES':'ACTIVITY'], 
                          inactiveDF.sample(n=len(activeDF)).loc[:, 'SMILES':'ACTIVITY']])
else:
    filtered = pd.concat([activeDF.sample(n=len(inactiveDF)).loc[:, 'SMILES':'ACTIVITY'], 
                          inactiveDF.loc[:, 'SMILES':'ACTIVITY']])

filtered.to_csv(f'{sys.argv[2]}_filtered_dataset.csv', index=False)

print('Filtered dataset created')

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import os
import sys
from SmilesPE.pretokenizer import atomwise_tokenizer


NC = '\033[0m'
RED = '\033[0;31m'

if len(sys.argv) < 3:
    print(f'{RED}Too few args{NC}')
    sys.exit()

df = pd.read_csv(f'{sys.argv[1]}', low_memory=False)

df = df[['SMILES', 'ACTIVITY']]

# check type, if numeric skip this and say youre skipping this
# if string proceed as normal
# if neither specify that this is an unknown datatype that cannot be used

if is_string_dtype(df['ACTIVITY']):
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
elif is_numeric_dtype(df['ACTIVITY']):
    print('Numeric dataset found, skipping boolean based filtration')

    df = df[~df['SMILES'].isnull()]

    df.to_csv(f'{sys.argv[2]}_filtered_dataset.csv', index=False)  
else:
    print(f'{RED}Dataset of unknown type found, must either be all strings of "Active" or "Inactive" or all numeric values{NC}')
    sys.exit(1)

print('Filtered dataset created')

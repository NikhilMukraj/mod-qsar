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

df = df[['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME']]

df = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Inconclusive', :]
df = df[~df['PUBCHEM_EXT_DATASOURCE_SMILES'].isnull()]

activeDF = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active', :]
inactiveDF = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Inactive', :]

if len(activeDF) <= len(inactiveDF):
    filtered = pd.concat([activeDF.loc[:, 'PUBCHEM_EXT_DATASOURCE_SMILES':'PUBCHEM_ACTIVITY_OUTCOME'], 
                          inactiveDF.sample(n=len(activeDF)).loc[:, 'PUBCHEM_EXT_DATASOURCE_SMILES':'PUBCHEM_ACTIVITY_OUTCOME']])
else:
    filtered = pd.concat([activeDF.sample(n=len(inactiveDF)).loc[:, 'PUBCHEM_EXT_DATASOURCE_SMILES':'PUBCHEM_ACTIVITY_OUTCOME'], 
                          inactiveDF.loc[:, 'PUBCHEM_EXT_DATASOURCE_SMILES':'PUBCHEM_ACTIVITY_OUTCOME']])

filtered.to_csv(f'{sys.argv[2]}_filtered_dataset.csv', index=False)

print('Filtered dataset created')

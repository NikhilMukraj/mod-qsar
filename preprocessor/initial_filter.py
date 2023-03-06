import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from SmilesPE.pretokenizer import atomwise_tokenizer


# df = pd.read_csv([i for i in os.listdir() if '.csv' in i][0])
if len(sys.argv) < 2:
    print('Too few args')
    sys.exit()

df = pd.read_csv(f'{sys.argv[1]}', low_memory=False)
df = df.drop(list(range(0,4)))

df = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] != 'Inconclusive', :]
df = df[~df['PUBCHEM_EXT_DATASOURCE_SMILES'].isnull()]

# all_tokens = [atomwise_tokenizer(i) for i in df['PUBCHEM_EXT_DATASOURCE_SMILES']]
# tokens = [i for sublist in all_tokens for i in sublist]
# tokens = list(set(tokens))
# tokens.sort()

# print(tokens, len(tokens))

# check if any new tokens

# if 'vocab.csv' not in os.listdir():
#     tokens_df = pd.DataFrame(tokens, columns=["tokens"])
#     tokens_df.to_csv(f'{os.getcwd()}//vocab.csv', index=None)
# else:
#     original_tokens = pd.read_csv(f'{os.getcwd()}//vocab.csv')['tokens']
#     original_tokens = original_tokens.to_list()
#     if tokens != original_tokens:
#         if bool(sys.argv[3]):
#             unwanted_tokens = [i for i in tokens if i not in original_tokens]
#             indices = [n for n, i in enumerate(all_tokens) if len(list(set(i) - set(original_tokens))) != 0]
#             df.drop(indices)
#             print(f'Amount of entries removed due to override: {len(indices)}')
#         else:
#             raise Exception('New tokens found')

activity_dict = {'Active': [], 'Inactive': []}
outcomes = df['PUBCHEM_ACTIVITY_OUTCOME'].to_list()
scores = df['PUBCHEM_ACTIVITY_SCORE'].to_list()
for i in range(len(df)):
    activity_dict[outcomes[i]].append(scores[i])

# execute if debug
# print(np.average(activity_dict['Inactive']), np.std(activity_dict['Inactive']))
# print(np.average(activity_dict['Active']), np.std(activity_dict['Active']))

activeDF = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Active', :]
inactiveDF = df.loc[df['PUBCHEM_ACTIVITY_OUTCOME'] == 'Inactive', :]

filtered = pd.concat([activeDF.iloc[:, 3:5], inactiveDF.sample(n=len(activeDF)).iloc[:, 3:5]])
filtered.to_csv(f'{os.getcwd()}//{sys.argv[2]}_filtered_dataset.csv', index=False)

print('Filtered dataset created')
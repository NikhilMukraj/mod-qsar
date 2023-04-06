from rdkit import Chem
from rdkit.Chem import Draw
from string_ga import calculateScore
from IPython.display import SVG
import pubchempy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import pubchempy
from concurrent.futures import ThreadPoolExecutor
import sys
import os


files = pd.read_csv(sys.argv[1]).iloc[:, 0]
path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists(f'{path}/generated_drugs/images'):
    os.makedirs(f'{path}/generated_drugs/images')

def mol_to_img(string, name):
    mol = Chem.MolFromSmiles(string)
    d2d = Draw.MolDraw2DCairo(1000, 1000)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()

    with open(f'{path}/generated_drugs/images/{name}_{int(calculateScore(mol))}.png', 'wb') as png_file:
        png_file.write(png_data)

df = pd.concat([pd.read_csv(i) for i in files])
df.reset_index(drop=True, inplace=True)

sanitized = [Chem.MolToSmiles(Chem.MolFromSmiles(i, sanitize=True)) for i in df['string']]

print('Writing images...')
for n, i in tqdm(enumerate(df['string'])):
    mol_to_img(Chem.MolToSmiles(Chem.MolFromSmiles(i, sanitize=True)), n)

def get_name(name):
    try:
        return pubchempy.get_compounds(i, namespace='smiles')
    except Exception:
        return None

if len(sys.argv) > 2:
    print('Getting compounds...')
    with ThreadPoolExecutor(max_workers=20) as executor:
        # compounds = [(get_name(i), i) for i in tqdm(sanitized)]
        compounds = executor.map(lambda i: (get_name(i), i), sanitized)

    compounds = [i[0][0].iupac_name for i in compounds if i[0] and i[0][0].iupac_name]

    if compounds:
        print('Writing names...')
        compounds_df = pd.DataFrame(np.array([[i[0] for i in compounds], [i[1] for i in compounds]]).tranpose(), 
                            columns=['names', 'strings'])
        compounds_df.to_csv(sys.argv[2])
    else:
        print('No compounds found')
from rdkit import Chem
from rdkit.Chem import Draw
from string_ga import calculateScore
from IPython.display import SVG
import pubchempy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pubchempy
import sys
import os


files = pd.read_csv(sys.argv[1]).iloc[:, 0]
path = os.path.dirname(os.path.abspath(__file__))

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

for n, i in enumerate(df['string']):
    mol_to_img(Chem.MolToSmiles(Chem.MolFromSmiles(i, sanitize=True)), n)

def get_name(name):
    try:
        return pubchempy.get_compounds(i, namespace='smiles')
    except Exception:
        return None

compounds = [(get_name(i), i) for i in sanitized]
compounds = [i[0][0].iupac_name for i in compounds if i[0] and i[0][0].iupac_name]

if compounds:
    compounds_df = pd.DataFrame(np.array([[i[0] for i in compounds], [i[1] for i in compounds]]).tranpose(), 
                        columns=['names', 'strings'])
    compounds_df.to_csv(sys.argv[2])
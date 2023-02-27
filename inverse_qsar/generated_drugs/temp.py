from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import py3Dmol
import numpy as np
import pandas as pd
import json
import sys
import os

def get_lipinski(string):
    molecule = Chem.MolFromSmiles(string)
    lipi = 0

    h_bond_donors = Lipinski.NumHDonors(molecule)
    h_bond_acceptors = Descriptors.NumHAcceptors(molecule)
    molecular_weight = Descriptors.ExactMolWt(molecule)
    logp = Descriptors.MolLogP(molecule)

    if h_bond_donors <= 5:
        lipi += .25
    if h_bond_acceptors <= 10:
        lipi += .25
    if 200 <= molecular_weight <= 500:
        lipi += .25
    if logp <= 5:
        lipi += .25

    return lipi
import pubchempy


with open('ssris.txt', 'r') as f:
    contents = f.read().split()

smiles = [pubchempy.get_compounds(i, 'name')[0].isomeric_smiles for i in contents]

with open('ssris.smi', 'w+') as f:
    for i in smiles:
        f.write(f'{i}\n')

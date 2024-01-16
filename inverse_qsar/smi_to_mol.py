import rdkit.Chem as Chem
from tqdm import tqdm
import os
import sys


GREEN = '\033[1;32m'
NC = '\033[0m'
RED = '\033[0;31m'

if len(sys.argv) < 3:
    print(f'{RED}Requires input filename and output directory name{NC}')
    sys.exit(1)

with open(sys.argv[1], 'r') as f:
    strings = f.read()

mols = [Chem.MolFromSmiles(i) for i in strings.split('\n') if i != '']
os.mkdir(sys.argv[2])

for n, i in tqdm(enumerate(mols)):
    Chem.MolToMolFile(i, f'{sys.argv[2]}/mol_{n}.mol')

plural = 's' if len(mols) > 1 else ''
print(f'{GREEN}Finished writing {len(mols)} file{plural}{NC}')

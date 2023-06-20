import pandas as pd
import sys


RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

if len(sys.argv) < 2:
    print(f'{RED}Requires filename as argument{NC}')

symbols = ['C', 'O', '(', '=', ')', '[C@@H]', '[C@H]', 'H', '1', 'N', 
            '2', '3', 'F', 'S', 'Cl', '#', '+', '-', '/', '4', 'B', 
            'Br', '\\', '5', 'I']

vocab_df = pd.DataFrame(symbols, columns=["tokens"])
vocab_df.to_csv(sys.argv[1], index=None)

print(f'{GREEN}Generated default vocab file with filename "{sys.argv[1]}"{NC}')

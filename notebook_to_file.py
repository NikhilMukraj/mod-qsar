import os
import sys
import json


if len(sys.argv) <= 2:
    print('Too few arguments')
    sys.exit()
elif '.ipynb' not in sys.argv[1]:
    print(f'{sys.argv[1]} not a notebook')
    sys.exit()

with open(f'{os.getcwd()}\\{sys.argv[1]}', 'r') as f:
    notebook = dict(json.loads(f.read()))

code = ''
for i in notebook['cells']:
    current_cell = dict(i)
    if current_cell['cell_type'] == 'code':
        code += ''.join(current_cell['source'])
        code += '\n\n'

code = code.strip()

with open(f'{os.getcwd()}\\{sys.argv[2]}', 'w+') as f:
    f.write(code)
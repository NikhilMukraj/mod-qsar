import sys
import argparse
import json
import chembl_backbone as chembl
import subprocess
from itertools import chain


GREEN = '\033[1;32m'
NC = '\033[0m'
RED='\033[0;31m'

parser = argparse.ArgumentParser(description='Add datasets to pre-existing vocab file using CHEMBL targets')
parser.add_argument('args')
parser.add_argument('-a', '--aggregate')
parser.add_argument('-n', '--number-of-augmentations', help='Number of augmentations to use when generating .npy files, defaults to 0')
parser.add_argument('-m', '--max-length', help='Maximum length of amount of tokens in each sample or false if none is necessary')
parser.add_argument('-o', '--override', help='Skip over strings with tokens not in vocab file')
parser.add_argument('-v', '--vocab', help='(Optional) filename of vocabulary file to use')
parser.add_argument('-s', '--sysimage', help='Use Julia --sysimage to run Julia component')

parsed_args = parser.parse_args()

if parsed_args.args[-5:] != '.json':
    print(f'{RED}{parsed_args.args} is not a .json file{NC}')
    sys.exit(1)

if parsed_args.aggregate and parsed_args.aggregate[-5:] != '.json':
    print(f'{RED}{parsed_args.aggregate} is not a .json file{NC}')
    sys.exit(1)

with open(parsed_args.args, 'r') as f:
    args = json.load(f)

aggregate_args = None
if parsed_args.aggregate:
    with open(parsed_args.aggregate, 'r') as f:
        aggregate_args = json.load(f)

if parsed_args.max_length and parsed_args.max_length.lower() != 'false':
    try:
        max_len = int(parsed_args.max_length)
    except ValueError:
        print(f'{RED}"max-length" must be a integer that is 0 or greater or false{NC}')
        sys.exit(1)

    if max_len < 0:
        print(f'{RED}"max-length" must be a integer that is 0 or greater or false{NC}')
        sys.exit(1)
elif parsed_args.max_length and parsed_args.max_length.lower() == 'false':
    max_len = False
elif parsed_args.max_length is None:
    max_len = False
else:
    print(f'{RED}"max-length" must be a integer that is 0 or greater or false{NC}')
    sys.exit(1)

if parsed_args.number_of_augmentations:
    try:
        num = int(parsed_args.number_of_augmentations)
    except ValueError:
        print(f'{RED}"number-of-augmentations" must be a integer that is 0 or greater{NC}')
        sys.exit(1)

    if num < 0:
        print(f'{RED}"number-of-augmentations" must be a integer that is 0 or greater{NC}')
        sys.exit(1)
else:
    num = 0

if not parsed_args.vocab:
    vocab = 'vocab.csv'
else:
    vocab = parsed_args.vocab

bool_dict = {'true' : True, 'false' : False}

if parsed_args.override and parsed_args.override.lower() not in ['false', 'true']:
    print(f'{RED}"override" must be a boolean{NC}')
    sys.exit(1)
elif parsed_args.override and parsed_args.override.lower() in bool_dict:
    override = bool_dict[parsed_args.override.lower()]
elif parsed_args.override is None:
    override = False

if parsed_args.sysimage and parsed_args.sysimage.lower() not in ['false', 'true']:
    print(f'{RED}"sysimage" must be a boolean{NC}')
    sys.exit(1)
elif parsed_args.sysimage and parsed_args.sysimage.lower() in bool_dict:
    sysimage = bool_dict[parsed_args.sysimage.lower()]
elif parsed_args.sysimage is None:
    sysimage = False

files_to_use = list(args.keys())
if aggregate_args:
    for name, value in aggregate_args.items():
        files_to_use.append(name)
        for filename in aggregate_args[name]['filenames']:
            files_to_use.remove(filename)

args_dict = {}
for filename in files_to_use:
    if filename in args:
        args_dict[filename] = args[filename]['tag']
    if aggregate_args and filename in aggregate_args:
        args_dict[filename] = aggregate_args[filename]['tag']

args_list = ['bash', './add_dataset.sh']
for i in args_dict.keys():
    args_list += ['-f', i]
for i in args_dict.values():
    args_list += ['-t', i]

if len([i for i in args_list if i == '-f']) > 1:
    print(f'{RED}Currently can only add one CHEMBL dataset at a time{NC}')
    sys.exit(1)

args_list += ['-n', str(num), '-m', str(max_len).lower(), '-o', str(override).lower(), 
              '-v', vocab, '-s', str(sysimage).lower()]

chembl.generate_dataset(args, aggregate_args=aggregate_args, do_full_processing=True)

with subprocess.Popen(args_list, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as process:
    for line in process.stdout:
        print(line, end='') 

if process.returncode != 0:
    raise subprocess.CalledProcessError(process.returncode, process.args)

print(f'{GREEN}Finished generating CHEMBL dataset with pre-existing vocab file{NC}')

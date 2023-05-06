import sys
import argparse
import json
import chembl_backbone as chembl
import subprocess
from itertools import chain


GREEN = '\033[1;32m'
NC = '\033[0m'
RED='\033[0;31m'

parser = argparse.ArgumentParser(description='Generate entire datasets using CHEMBL targets')
parser.add_argument('args')
parser.add_argument('-a', '--aggregate', help='Aggregate datasets into singular files')
parser.add_argument('-f', '--full-preprocessing', help='Generate complete dataset with .npy files')
parser.add_argument('-n', '--number-of-augmentations', help='Number of augmentations to use when generating .npy files, defaults to 0')

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

if parsed_args.full_preprocessing:
    if parsed_args.full_preprocessing.lower() == 'true':
        do_full_processing = True
    elif parsed_args.full_preprocessing.lower() == 'false':
        do_full_processing = False
    else:
        print(f'{RED}"full-preprocessing" must be a boolean{NC}')
        sys.exit(1)
else:
    do_full_processing = False

if do_full_processing and parsed_args.number_of_augmentations:
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

chembl.generate_dataset(args, aggregate_args=aggregate_args, do_full_processing=do_full_processing)

if not do_full_processing:
    sys.exit(0)

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

args_list = ['bash', './preprocessor.sh']
for i in args_dict.keys():
    args_list += ['-f', i]
for i in args_dict.values():
    args_list += ['-t', i]

args_list += ['-n', str(num)]

process = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
process.wait()
output, error_code = process.communicate()

if process.returncode != 0:
    print(f'{RED}Process failed, error:{NC}')
    print(process.stderr.read().decode().strip())
    sys.exit(1)

print(f'{GREEN}Finished generating dataset{NC}')
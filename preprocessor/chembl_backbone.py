from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import json
from itertools import chain


GREEN = '\033[1;32m'
NC = '\033[0m'
RED='\033[0;31m'

necessary_args = {
    'target_chembl_id' : [str], 
    'activity_type' : [str], 
    'min' : [float, int], 
    'max' : [float, int],
}

optional_args = {
    'tag' : [str],
}


def get_activities(target, activity_type, activity, name=None):
    if name:
        print(f'Getting bioactivty for {name}')

    activity_set = [i for i in tqdm(activity.filter(target_chembl_id=target).filter(standard_type=activity_type))]

    if name:
        print(f'{GREEN}Finished getting bioactivity for {name}{NC}')

    return activity_set


def convert_to_nm(arr):
    unit = arr[2]
    val = arr[1]

    if unit == 'µM':
        val *= 1000
    
    arr[1], arr[2] = val, 'nM'

    return arr

def generate_dataset(args, aggregate_args=None, do_full_processing=False):
    for name, i in args.items():
        for arg in i:
            if arg not in necessary_args and arg not in optional_args:
                print(f'{RED}Unknown argument at activity query "{name}": {arg}{NC}')
                sys.exit(1)

    if do_full_processing:
        for name, i in args.items():
            if 'tag' not in i:
                print(f'{RED}Full preprocessing requires "tag" arguments, "tag" not found in activity query "{name}"{NC}')
                sys.exit(1)

    if aggregate_args and do_full_processing:
        for name, i in aggregate_args.items():
            if 'tag' not in i:
                print(f'{RED}Full preprocessing requires "tag" arguments, "tag" not found in aggregate "{name}"{NC}')
                sys.exit(1)

    for i in necessary_args:
        for name, arg_set in args.items():
            if i not in arg_set and i not in optional_args:
                print(f'{RED}Missing argument in activity query "{name}": {i}{NC}')
                sys.exit(1)

    for name, i in args.items():
        for key, value in i.items():
            if key in necessary_args and type(value) not in necessary_args[key]:
                print(f'{RED}Type mismatch in argument at activity query "{name}", expected type in {necessary_args[key]} at "{key}" argument but got {type(value)}{NC}')
                sys.exit(1)

            if key in optional_args and type(value) not in optional_args[key]:
                print(f'{RED}Type mismatch in argument at activity query "{name}", expected type in {optional_args[key]} at "{key}" argument but got {type(value)}{NC}')
                sys.exit(1)

            if key == 'min' and (i[key] < 0 or i[key] >= i['max']):
                print(f'{RED}"min" argument in activity query "{name}" must be greater than 0 and less than argument "max"{NC}')
                sys.exit(1)

            if key == 'max' and i[key] <= i['min']:
                print(f'{RED}"max" argument in activity query "{name}" must be greater than argument "max"{NC}')
                sys.exit(1)

    if len(set(i['tag'] for i in args.values())) != len(args.keys()):
        print(f'{RED}Activity query cannot have duplicate "tag" arguments{NC}')
        sys.exit(1)

    necessary_aggregate_args = {
        'filenames' : [list],
    }

    if aggregate_args:
        for key, entry in aggregate_args.items():
            for i in entry.keys():
                if i not in necessary_aggregate_args and i not in optional_args:
                    print(f'{RED}Unknown argument in aggregate "{key}": {i}{NC}')
                    sys.exit(1)
                
            for value in entry:
                if value in necessary_aggregate_args and type(entry[value]) not in necessary_aggregate_args[value]:
                    print(f'{RED}Type mismatch in argument at aggregate "{key}", expected type in {necessary_aggregate_args[value]} at "{value}" argument but got {type(entry[value])}{NC}')
                    sys.exit(1)
                    
                if value in optional_args and type(entry[value]) not in optional_args[value]:
                    print(f'{RED}Type mismatch in argument at aggregate "{key}", expected type in {optional_args[value]} at "{value}" argument but got {type(entry[value])}{NC}')
                    sys.exit(1)

            if any(type(filename) != str for filename in aggregate_args[key]['filenames']):
                print(f'{RED}Type mismatch in argument at aggregate "{key}", expected {str} in "filenames" argument')
                sys.exit(1)

            for filename in entry['filenames']:
                if filename not in args:
                    print(f'{RED}"{filename}" not found in {sys.argv[1]}{NC}')
                    sys.exit(1)

        if len(set(i['tag'] for i in aggregate_args.values())) != len(aggregate_args.keys()):
            print(f'{RED}Aggregate cannot have duplicate "tag" arguments{NC}')
            sys.exit(1)    

    # target = new_client.target
    activity = new_client.activity

    targets = [i['target_chembl_id'] for i in args.values()]
    names = [i for i in args.keys()]
    activity_types = [i['activity_type'] for i in args.values()]

    activities = {(name, target) : get_activities(target, activity_type, activity, name=name) 
                for name, target, activity_type in zip(names, targets, activity_types)}

    # units = set(chain.from_iterable([[i['standard_units'] for i in activity_set if i['standard_units']] for activity_set in activities.values()]))

    activity_dict = {}

    for key, activity_set in activities.items():
        name, _ = key
        activity_dict[name] = [convert_to_nm([i['canonical_smiles'], float(i['value']), i['standard_units']]) for i in activity_set 
                                if i['standard_units'] not in [None, 'ug.mL-1'] and i['value'] is not None]

    dfs = {}

    for name, activity_set in activity_dict.items():
        is_active = lambda val: args[name]['min'] <= val <= args[name]['max']

        df = pd.DataFrame([], columns=['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME'])
        for i in activity_set:
            smiles_string, value, _ = i
            df.loc[len(df.index)] = [smiles_string, 'Active' if is_active(value) else 'Inactive']

        dfs[name] = df

    [df.to_csv(name, index=None) for name, df in dfs.items()]

    for name, df in dfs.items():
        print(name)
        print(df['PUBCHEM_ACTIVITY_OUTCOME'].value_counts())

    if aggregate_args:
        for filename, names in aggregate_args.items():
            names = names['filenames']
            pd.concat([dfs[name] for name in names]).to_csv(filename, index=None)

    print(f'{GREEN}Finished creating CHEMBL dataset{NC}')
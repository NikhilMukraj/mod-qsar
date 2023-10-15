from chembl_webresource_client.new_client import new_client
from tqdm import tqdm
import pandas as pd
import sys
import json
import requests
import xmltodict
# from itertools import chain


GREEN = '\033[1;32m'
NC = '\033[0m'
RED = '\033[0;31m'

# list out all arguments used
necessary_args = {
    'id' : [str], 
    'activity_type' : [str],
}

boolean_activity_args = {
    'min' : [float, int], 
    'max' : [float, int],
}

optional_args = {
    'tag' : [str],
}

necessary_aggregate_args = {
    'filenames' : [list],
}

def check_arg_with_type(arg_type, given_arg, args_template):
    if arg_type in args_template and type(given_arg) not in args_template[arg_type]:
        print(f'{RED}Type mismatch in argument at activity query "{arg_type}", expected type in {args_template[arg_type]} at "{arg_type}" argument but got {type(given_arg)}{NC}')
        sys.exit(1)

def get_activities(target, activity_type, activity, name=None):
    if name:
        print(f'Getting bioactivty for {name}')

    # processes chembl ids versus uniprot
    if target.startswith('CHEMBL'):
        activity_set = [i for i in tqdm(activity.filter(target_chembl_id=target).filter(standard_type=activity_type))]

        if len(activity_set) == 0:
            print(f'{RED}No bioactivities found{NC}')
            sys.exit(1)

        if name:
            print(f'{GREEN}Finished getting bioactivity for {name}{NC}')
    elif target.startswith('P'):
        ligands_link = f'https://bindingdb.org/axis2/services/BDBService/getLigandsByUniprot?uniprot={target}'
        response = requests.get(ligands_link)
        response_dict = xmltodict.parse(response.text)

        activity_set = []

        for i in tqdm(response_dict['bdb:getLigandsByUniprotResponse']['bdb:affinities']):
            if i['bdb:affinity_type'] in activity_type \
                and '<' not in i['bdb:affinity'] \
                and '>' not in i['bdb:affinity']:
                activity_set.append({
                    'canonical_smiles' : i['bdb:smiles'].split()[0],
                    'value' : float(i['bdb:affinity']),
                    'standard_units' : 'nM',
                })

        if name:
            print(f'{GREEN}Finished getting bioactivity for {name}{NC}')
    else:
        print(f'{RED}Database "{db}" is unknown{NC}')
        sys.exit(1)

    return activity_set


def convert_to_nm(arr):
    unit = arr[2]
    val = arr[1]

    if unit == 'nM':
        return arr
    if unit == 'µM':
        val *= 1000
    
    arr[1], arr[2] = val, 'nM'

    return arr

def generate_dataset(args, aggregate_args=None, do_full_processing=False):
    for name, i in args.items():
        for arg in i:
            if arg not in necessary_args \
                and arg not in optional_args \
                and arg not in boolean_activity_args:
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

    boolean_activity = {}
    for name, arg_set in args.items():
        boolean_arg_set = [i in arg_set for i in boolean_activity_args.keys()]
        if len(boolean_arg_set) == 1:
            print(f'{RED}Activity query must specify both a "min" and "max" argument or neither{NC}')
            sys.exit(1)
            
        boolean_activity[name] = all(boolean_arg_set)

    for name, i in args.items():
        for key, value in i.items():
            check_arg_with_type(key, value, necessary_args)

            check_arg_with_type(key, value, optional_args)

            if key == 'min' and (i[key] < 0 or i[key] >= i['max']):
                print(f'{RED}"min" argument in activity query "{name}" must be greater than 0 and less than argument "max"{NC}')
                sys.exit(1)

            if key == 'max' and i[key] <= i['min']:
                print(f'{RED}"max" argument in activity query "{name}" must be greater than argument "max"{NC}')
                sys.exit(1)

    if len(set(i['tag'] for i in args.values())) != len(args.keys()):
        print(f'{RED}Activity query cannot have duplicate "tag" arguments{NC}')
        sys.exit(1)

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

    activity_client = new_client.activity
    target_client = new_client.target

    targets = [i['id'] for i in args.values()]
    for i in targets:
        if i.startswith('CHEMBL'):
            if (target_return := target_client.filter(chembl_id=i).only(['id', 'pref_name'])):
                print(f'{GREEN}Found CHEMBL target for {i}: {target_return[0]["pref_name"]}{NC}')
            else:
                print(f'{RED}Cannot find corresponding CHEMBL target for "{i}"{NC}')
                sys.exit(1)
        elif i.startswith('P'):
            name_link = f'https://rest.uniprot.org/uniprotkb/{i}.json'
            response = requests.get(name_link)
            name_json = json.loads(response.text)

            if 'proteinDescription' not in name_json:
                print(f'{RED}Cannot find corresponding BindingDB target for "{i}"{NC}')
                sys.exit(1)

            target_name = name_json['proteinDescription']['recommendedName']['fullName']['value']
            print(f'{GREEN}Found BindingDB target for {i}: {target_name}{NC}')
        else:
            print(f'{RED}Target input must be either valid CHEMBL ID or UniProt ID{NC}')
            sys.exit(1)

    names = [i for i in args.keys()]
    activity_types = [i['activity_type'] for i in args.values()]

    activities = {(name, target) : get_activities(target, activity_type, activity_client, name) 
                for name, target, activity_type in zip(names, targets, activity_types)}

    # units = set(chain.from_iterable([[i['standard_units'] for i in activity_set if i['standard_units']] for activity_set in activities.values()]))

    activity_dict = {}

    for key, activity_set in activities.items():
        name, _ = key
        activity_dict[name] = [convert_to_nm([i['canonical_smiles'], float(i['value']), i['standard_units']]) for i in activity_set 
                                if i['standard_units'] not in [None, 'ug.mL-1'] and i['value'] is not None]

    dfs = {}

    for name, activity_set in activity_dict.items():
        df = pd.DataFrame([], columns=['SMILES', 'ACTIVITY'])
        if boolean_activity[name]:
            is_active = lambda val: args[name]['min'] <= val <= args[name]['max']

            for i in activity_set:
                smiles_string, value, _ = i
                df.loc[len(df.index)] = [smiles_string, 'Active' if is_active(value) else 'Inactive']
        else:
            for i in activity_set:
                smiles_string, value, _ = i
                df.loc[len(df.index)] = [smiles_string, value]
        
        dfs[name] = df

    [df.to_csv(name, index=None) for name, df in dfs.items()]

    for name, df in dfs.items():
        print(name)
        if boolean_activity[name]:
            print(df['ACTIVITY'].value_counts())
        else:
            print(f'Binding affinity average (nM): {df["ACTIVITY"].mean()}')
            print(f'Binding affinity standard deviation (nM): {df["ACTIVITY"].std()}')

    if aggregate_args:
        for filename, names in aggregate_args.items():
            names = names['filenames']
            if len(set([boolean_activity[i] for i in names])) == 1:
                pd.concat([dfs[name] for name in names]).to_csv(filename, index=None)
            else:
                print(f'{RED}Aggregate of {names} should have either all boolean activities or all actual binding affinities{NC}')
                sys.exit()

        print(f'{GREEN}Finished aggregation of dataset{NC}')        

    print(f'{GREEN}Finished creating dataset{NC}')
    
# Mod-QSAR

A modular inverse QSAR pipeline

## Overview

Built and tested on WSL using Ubuntu 22.04.1 LTS, currently being tested on Ubuntu without WSL.
Built using [SmilesEnumerator](https://github.com/EBjerrum/SMILES-enumeration), [StringGA](https://github.com/jensengroup/String-GA), Python and Julia.
The pipeline works by first taking in a series of PubChem or CHEMBL `.csv` files that contain a SMILES string associated with a biotarget.
The pipeline then filteres that dataset such that an equal amount of active and inactive compounds are found within the dataset.
The pipeline starting augmenting the dataset by enumerating over SMILES strings and generating a vocabulary of tokens used in those `.csv` files.
After this vocabulary is generated the strings are converted into a series of onehot encoded arrays dumped into a `.npy` file.
For each biotarget given in the initial preprocessing phase, a QSAR model is trained to determine whether a compound is active or inactive and the models are saved as `.h5` files.
The accuracy of these models can be increased by predicting on a series of augmented strings and then taking the average prediction on those strings.
The optimal amount of strings to augment is calculated by finding the carrying capacity of a differential equation and finding where the model's prediction accuracy reaches that carrying capacity as the amount of augmentations is increased.
After the augmentation hyperparameter is optimized, a genetic algorithm is used to mutate a series of initial existing chemicals and optimizing a series of drug-likeness measures and given QSAR models.

## How To

(This project is recommended to run within a virtual environment)

Download dependencies using:

```bash
bash ./initialize.sh
```

(Note that the first run may take a little while since it needs to compile the necessary files but all subsequent runs should be faster)

Preprocess datasets using:

```bash
cd preprocessor
bash ./preprocessor.sh -f dataset1.csv -f dataset2.csv -t tag1 -t tag2 -n 10 -v vocab.csv -s true
```

- `-f` : PubChem bioassay `.csv` file, multiple can be specified
- `-t` : String specifying a tag to add to the preprocessed dataset
- `-n` : Positive integer representing amount of augmentations to add
- `-v` : (Optional) filename of vocabulary file, defaults to `vocab.csv`
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

To use a non PubChem dataset, use a `.csv` file in the following format:

| PUBCHEM_EXT_DATASOURCE_SMILES    | PUBCHEM_ACTIVITY_OUTCOME |
|----------------------------------|--------------------------|
| CC1=NN(C2=NC(=O)N(C(=O)C2=N1)C)C | Active                   |
| CC1=C2C(=NN1)C(=S)NC(=O)N2       | Inactive                 |
| ...                              | ...                      |

Add another dataset using a previously generated `vocab.csv`:

```bash
cd preprocessor
bash ./add_dataset.sh -f dataset1.csv -t tag1 -n 10 -m 196 -o true -v vocab.csv -s true
```

- `-f` : PubChem bioassay `.csv` file, multiple can be specified
- `-t` : String specifying a tag to add to the preprocessed dataset, amount of `-t` and `-f` arguments must be the same
- `-n` : Positive integer representing amount of augmentations to add
- `-m` : Maximum length of tokens, any samples found that are longer are removed from the dataset
- `-o` : Boolean representing whether to ignore or override tokens not found in initial vocabulary
- `-v` : (Optional) filename representing vocabulary file to use (defaults to `vocab.csv`)
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

Curate datasets using CHEMBL:

```bash
cd preprocessor
python3 chembl_dataset_generator.py dataset_args.json -a aggregate_args.json -f true -n 10 -v vocab.csv -s true
```

- First argument : `.json` file that specifies the target and threshold for activity
- `-a` : Aggregate datasets into singular files
- `-f` : Boolean that states whether to generate `.npy` files in same manner as `preprocessor.sh` script
- `-n` : Integer amount greater than or equal to 0 representing amount of augmentations to include in `preprocessor.sh` script
- `-v` : (Optional) filename of vocabulary file, defaults to `vocab.csv`
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

First `.json` file arguments:

- Filename
  - `target_chembl_id` : Valid target ID from CHEMBL database
  - `activity_type` : Valid type of activity from CHEMBL database, (`IC50` or `EC50` for example)
  - `tag` : String representing tag to use in `preprocessor.sh` script
  - `min` : Float minimum threshold for being considered active
  - `max` : Float maximum threshold for being considered active

Example `dataset_args.json`:

```json
{
    "serotonin_antagonist.csv" : {
        "target_chembl_id" : "CHEMBL224",
        "activity_type" : "IC50",
        "tag" : "sero",
        "min" : 0,
        "max" : 100
    },
    "d2_antagonist.csv" : {
        "target_chembl_id" : "CHEMBL217",
        "activity_type" : "IC50",
        "tag": "d2",
        "min" : 0,
        "max" : 100
    },
    "d3_antagonist.csv" : {
        "target_chembl_id" : "CHEMBL234",
        "activity_type" : "IC50",
        "tag" : "d3",
        "min" : 0,
        "max" : 100
    }
}
```

Example `aggregate_args.json`:

```json
{
    "dopamine_antagonist.csv" : {
        "tag" : "dopa",
        "filenames": ["d2_antagonist.csv", "d3_antagonist.csv"]
    }
}
```

Add another dataset to a previously generated `vocab.csv` using CHEMBL (must be a singular dataset with or without aggregation):

```bash
cd preprocessor
python3 ./chembl_add_dataset.py dataset_args.json -a aggregate_args.json -n 10 -m 196 -o true -v vocab.csv -s true
```

- First argument: `.json` file that specifies the target and threshold for activity
- `-a` : Aggregate datasets into singular files
- `-n` : Positive integer representing amount of augmentations to add
- `-m` : Maximum length of tokens, any samples found that are longer are removed from the dataset
- `-o` : Boolean representing whether to ignore or override tokens not found in initial vocabulary
- `-v` : (Optional) filename of vocabulary to use (defaults to `vocab.csv`)
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

(See above examples for `dataset_args.json` and `aggregate_args.json`)

Train QSAR model:

```bash
cd predictor
python3 train_keras_rnn.py X.npy Y.npy 100 name testX.npy testY.npy
```

- First argument : A `.npy` file containing the tokenized X features
- Second argument : Must be labels in a `.npy` file
- Third argument : Amount of epochs to train for (positive integer), final argument specifies tag to name the model and training history
- Fourth argument : Name to add to model and model history outfiles
- Fifth argument : (Optional) name of file to dump features given dataset into for `optimize_n.sh`
- Sixth argument : (Optional) name of file to dump labels given dataset into for `optimize_n.sh`

(Optional) Optimize accuracy post training with additional augmentations:

```bash
cd predictor
bash ./optimize_n.sh -x testX.npy -y testY.npy -m rnn_model.h5 -v ../preprocessor/vocab.csv -s 10
```

- `-x` : `.npy` file containing tokenized X features
- `-y` : `.npy` file containing labels
- `-m` : Path of model to optimize
- `-v` : (Optional) filename of vocabulary to use (defaults to `../preprocessor/vocab.csv`)
- `-s` : (Optional) positive integer greater than 0, program will sample 1 in `-s` entries in `-x` and `-y` to evaluate

Generate chemicals:

```bash
cd inverse_qsar
python3 inverse_qsar_cli.py args.json chemicals_file.csv fitness_scores.csv
```

- First argument : A `.json` file containing all arguments to be used while generating
  - `population_size` : Initial size of chemicals pool
  - `mating_pool_size` : Size of genetic mating pool
  - `generations` : Number of iterations
  - `mutation_rate` : Chance that chemical will be changed randomly during training
  - `seed` : Random seed
  - `average_size` : Average size of chemical
  - `size_stdev` : Average standard deviation of chemical
  - `string_type` : Type of string formatting (recommended to use SMILES)
  - `scoring_function` : Type of scoring function to use
  - `strict` : Whether to completely eliminate molecules based on a weight threshold
  - `augment` : List containing boolean whether to augment data for model scoring function and integer how many times to augment
  - `max_len` : Maximum token length of molecules (automatically grab `max_len` by providing `null` as an argument)
  - `max_score` : Score to stop at
  - `prune_population` : Trim size of population
  - `target`: Target value for scoring functions to optimize
  - `weight`: Weight to apply to each output of scoring function
  - `file_name` : Pre-existing file of molecules to draw initial population from
  - `vocab` : Pre-existing file containing the vocabulary mapping
- Second argument : A `.csv` to dump the molecules into  
- Third argument : Optional `.csv` file to dump fitness scores after training

Example `args.json`:

```json
{
    "population_size" : 100,
    "mating_pool_size" : 100,
    "generations" : 20,
    "mutation_rate" : 0.05,
    "seed" : null,
    "average_size" : 375.0,
    "size_stdev" : 100.0,
    "string_type" : "SMILES",
    "scoring_function" : ["dopa_rnn_model.h5", "sero_rnn_model.h5", "custom_lipinski", "pains", "limit_rings"],
    "strict": true,
    "threads": 2,
    "augment": [true, 5],
    "max_len": 196,
    "max_score" : 1.0,
    "prune_population" : true,
    "target" : [1, 0, 1, 0, 1, 1, 1],
    "weight" : [1, 1, 1, 1, 1, 1, 1],
    "file_name" : "cl_f.smi",
    "vocab" : "../preprocessor/vocab.csv"
}
```

Postprocess generated `.csv` of molecular candidates into image files and check to see if any are already known:

```bash
cd inverse_qsar
python3 postprocessor.py ./generated_drugs/images files.csv names.csv
```

- First argument : Directory to write images to
- Second argument : `.csv` file containing `.csv` files generated by `inverse_qsar_cli.py` to be aggregated (also works with singular entry)
- Third argument : Optional argument containing name to write any chemicals that matched a known database

`files.csv` should be in the following format:

| files           |
|-----------------|
| chemicals_1.csv |
| chemicals_2.csv |
| ...             |

## Scoring Functions

- `model` : Uses the QSAR model as a scoring function
- `lipinski` : Uses Lipinski's Rule of Five
- `qed` : Uses QED drug-likeness measure
- `pains_filter` : Filters out PAINS
- `custom_lipinski` : Uses a custom weighted version of Lipinski's Rule of Five

## Todo

- Change preprocessing step to append to `.npy` file as loop progresses
- Range option for `optimize_n.sh`
- Debug information
- Add hyperparameter optimization
- Add Flux model integration

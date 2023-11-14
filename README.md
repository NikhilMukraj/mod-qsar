# Mod-QSAR

A modular inverse QSAR pipeline

## Overview

Built and tested on WSL using Ubuntu 22.04.1 LTS, has been tested for Ubuntu without WSL and works there as well.
Built using [SmilesEnumerator](https://github.com/EBjerrum/SMILES-enumeration), [StringGA](https://github.com/jensengroup/String-GA), Python (3.10.6) and Julia (1.8.1).
The pipeline works by first taking in a series of `.csv` files that contain a SMILES string associated with a biotarget binding outcome that can either be manually inputted or pulled from the [ChEMBL](https://www.ebi.ac.uk/chembl/) or [BindingDB](https://www.bindingdb.org/rwd/bind/index.jsp).
The pipeline then filters that dataset such that an equal amount of active and inactive compounds are found within the dataset.
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

Note, vocabulary file (examples use `vocab.csv`) must be a `.csv` file in the following format:

| tokens |
|--------|
| C      |
| F      |
| -      |
| 3      |
| ...    |

Preprocess datasets using:

```bash
cd preprocessor
bash ./preprocessor.sh -f dataset1.csv -f dataset2.csv -t tag1 -t tag2 -n 10 -v vocab.csv -s true
```

- `-f` : Bioassay `.csv` file, multiple can be specified
- `-t` : String specifying a tag to add to the preprocessed dataset
- `-n` : Positive integer representing amount of augmentations to add
- `-v` : (Optional) filename of vocabulary file, defaults to `vocab.csv`
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

To use a dataset, use a `.csv` file in the following format:

| SMILES                           | ACTIVITY                 |
|----------------------------------|--------------------------|
| CC1=NN(C2=NC(=O)N(C(=O)C2=N1)C)C | Active                   |
| CC1=C2C(=NN1)C(=S)NC(=O)N2       | Inactive                 |
| ...                              | ...                      |

Generate a default vocab file with default symbols:

```bash
cd preprocessor
python default_vocab_generator.py vocab.csv
```

- First argument : Filename of vocabulary file

Add another dataset using a previously generated `vocab.csv`:

```bash
cd preprocessor
bash ./add_dataset.sh -f dataset1.csv -t tag1 -n 10 -m 196 -o true -v vocab.csv -s true
```

- `-f` : Bioassay `.csv` file, multiple can be specified, use the `.csv` shown above
- `-t` : String specifying a tag to add to the preprocessed dataset, amount of `-t` and `-f` arguments must be the same
- `-n` : Positive integer representing amount of augmentations to add
- `-m` : Maximum length of tokens, any samples found that are longer are removed from the dataset
- `-o` : Boolean representing whether to ignore or override tokens not found in initial vocabulary
- `-v` : (Optional) filename representing vocabulary file to use (defaults to `vocab.csv`)
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

Curate datasets using ChEMBL:

```bash
cd preprocessor
python3 dataset_generator.py dataset_args.json -a aggregate_args.json -f true -n 10 -v vocab.csv -s true
```

- First argument : `.json` file that specifies the target and threshold for activity
- `-a` : Aggregate datasets into singular files
- `-f` : Boolean that states whether to generate `.npy` files in same manner as `preprocessor.sh` script
- `-n` : Integer amount greater than or equal to 0 representing amount of augmentations to include in `preprocessor.sh` script
- `-v` : (Optional) filename of vocabulary file, defaults to `vocab.csv`
- `-s` : (Optional) boolean as to whether or not to use a sysimage when running Julia component

First `.json` file arguments:

- Filename
  - `id` : Valid target ID (either input a valid ChEMBL target to pull from the ChEMBL database or a valid UniProt target to pull from BindingDB)
  - `activity_type` : Valid type of binding activity from ChEMBL or BindingDB, (`IC50` or `EC50` for example)
  - `tag` : String representing tag to use in `preprocessor.sh` script
  - `min` : Float minimum threshold for being considered active (in nM)
  - `max` : Float maximum threshold for being considered active (in nM)
    - `min` and `max` arguments only necessary for classification, for a regression dataset omit these arguments

Example `dataset_args.json`:

```json
{
    "serotonin_antagonist.csv" : {
        "id" : "CHEMBL224",
        "activity_type" : "IC50",
        "tag" : "sero",
        "min" : 0,
        "max" : 100
    },
    "other_serotonin_antagonist.csv" : {
        "id" : "P28223",
        "activity_type" : "IC50",
        "tag" : "sero_other",
        "min" : 0,
        "max" : 100
    },
    "d2_antagonist.csv" : {
        "id" : "CHEMBL217",
        "activity_type" : "IC50",
        "tag": "d2",
        "min" : 0,
        "max" : 100
    },
    "d3_antagonist.csv" : {
        "id" : "CHEMBL234",
        "activity_type" : "IC50",
        "tag" : "d3",
        "min" : 0,
        "max" : 100
    },
    "regression_d3_antagonist.csv" : {
        "id" : "P28223",
        "activity_type" : "Ki",
        "tag" : "d3_regression"
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

Add another dataset to a previously generated `vocab.csv` using ChEMBL (must be a singular dataset with or without aggregation):

```bash
cd preprocessor
python3 ./add_dataset.py dataset_args.json -a aggregate_args.json -n 10 -m 196 -o true -v vocab.csv -s true
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
bash ./optimize_n.sh -x testX.npy -y testY.npy -m rnn_model.h5 -v ../preprocessor/vocab.csv -s 10 -a 2 -b 11 -i 2 -p false
```

- `-x` : `.npy` file containing tokenized X features
- `-y` : `.npy` file containing labels
- `-m` : Path of model to optimize
- `-v` : (Optional) filename of vocabulary to use (defaults to `../preprocessor/vocab.csv`)
- `-f` : (Optional) filename of outfile (defaults to `augmented_accs.csv`)
- `-s` : (Optional) positive integer greater than 0, program will sample 1 in `-s` entries in `-x` and `-y` to evaluate
- `-a` : (Optional) positive integer greater than 0, minimum part of augmentation number range, defaults to 2
- `-b` : (Optional) positive integer greater than 0, maximum part of augmentation number range, defaults to 11
- `-i` : (Optional) positive integer greater than 0, increment of augmentation number range, defaults to 2
- `-p` : (Optional) boolean specifying whether or not to load packages with --sysimage, defaults to false

Generate chemicals:

```bash
cd inverse_qsar
python3 inverse_qsar_cli.py args.json chemicals_file.csv fitness_scores.csv
```

- First argument : A `.json` file containing all arguments to be used while generating
  - `population_size` : Initial size of chemicals pool (positive integer)
  - `mating_pool_size` : Size of genetic mating pool (positive integer)
  - `generations` : Number of iterations (positive integer)
  - `mutation_rate` : Chance that chemical will be changed randomly during training (float between 0 and 1)
  - `seed` : Random seed (null or an integer)
  - `average_size` : Average size of chemical (positive integer or float)
  - `size_stdev` : Average standard deviation of chemical (positive integer or float)
  - `string_type` : Type of string formatting (recommended to use `SMILES` as input or specify a filepath to a file in the same format as `vocab.csv`)
  - `scoring_function` : Type of scoring function to use
  - `strict` : Whether to completely eliminate molecules based on a weight threshold (boolean)
  - `augment` : List containing boolean whether to augment data for model scoring function and integer how many times to augment (positive integer)
  - `max_len` : Maximum token length of molecules (positive integer or automatically grab `max_len` by providing `null` as an argument)
  - `max_score` : Score to stop at (float)
  - `prune_population` : Trim size of population (boolean)
  - `target`: Target value for scoring functions to optimize (list of values between 0 and 1)
  - `weight`: Weight to apply to each output of scoring function (list of floats)
  - `file_name` : Pre-existing file of molecules to draw initial population from (filepath)
  - `vocab` : Pre-existing file containing the vocabulary mapping (filepath)
- Second argument : A `.csv` to dump the molecules into  
- Third argument : (Optional) `.csv` file to dump fitness scores after training

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
  - To use a regression model, prefix the model filenname with `regr`, the output will be scaled between 0 and 500 nM, the output should be 0 if the input was 0 nM and 1 if the input was 500 nM or more
- `lipinski` : Uses Lipinski's Rule of Five, (1 if true 0 otherwise)
- `qed` : Uses QED drug-likeness measure, (1 if true 0 otherwise)
- `ghose` : Uses Ghose drug-likeness measure, (1 if true 0 otherwise)
- `limit_rings` : Returns 0 if molecule has carbon rings larger than 6 atoms
- `pains_filter` : Filters out PAINS, (1 if no PAINS substructures found 0 otherwise)
- `custom_lipinski` : Uses a custom weighted version of Lipinski's Rule of Five that takes into account how synthesizable the molecule is, (1 if true 0 otherwise)
- `bbb_permeable` : Checks if molecule is blood brain barrier permeable using the [BOILED-egg method](https://github.com/bfmilne/pyBOILEDegg/tree/main), (1 if true 0 otherwise)
- `gastro_absorption` : Checks if molecule is has high gastrointestinal absorption using the [BOILED-egg method](https://github.com/bfmilne/pyBOILEDegg/tree/main), (1 if true 0 otherwise)

## Custom Scoring Functions

Use the format `"./filepath/to/python_file.py:function_name"` as an element in a list passed in the `scoring_function` argument where a `:` deliminates what is the filepath and what is the function name. The custom function must return a float. Custom functions must be specified last.

## Todo

- Add option for preprocessing step to append to `.npy` file as loop progresses for memory sake
- Debug information
- Add hyperparameter optimization
- Add Flux model integration

# Inverse QSAR Pipeline

## Overview

Built using [SmilesEnumerator](https://github.com/EBjerrum/SMILES-enumeration), [StringGA](https://github.com/jensengroup/String-GA), Python and Julia.
The pipeline works by first taking in a series of PubChem `.csv` files that contain a SMILES string associated with a biotarget. T
The pipeline then filteres that dataset such that an equal amount of active and inactive compounds are found within the dataset.
The pipeline starting augmenting the dataset by enumerating over SMILES strings and generating a vocabulary of tokens used in those .csv files.
After this vocabulary is generated the strings are converted into a series of onehot encoded arrays dumped into a numpy file.
For each biotarget given in the initial preprocessing phase, a QSAR model is trained to determine whether a compound is active or inactive and the models are saved as `.h5` files.
The accuracy of these models can be increased by predicting on a series of augmented strings and then taking the average prediction on those strings.
The optimal amount of strings to augment is calculated by finding the carrying capacity of a differential equation and finding where the model's prediction accuracy reaches that carrying capacity as the amount of augmentations is increased.
After the augmentation hyperparameter is optimized, a genetic algorithm is used to mutate a series of initial existing chemicals and optimizing a series of drug-likeness measures and given QSAR models.

## How To

Download dependencies using:

```bash
./initialize.sh
```

Preprocess datasets using:

```bash
./preprocessor/preprocessor.sh -f dataset1.csv -f dataset2.csv -t tag1 -t tag2 -n 10
```

- `-f` : PubChem bioassay .csv file, multiple can be specified
- `-t` : String specifying a tag to add to the preprocessed dataset
- `-n` : Positive integer representing amount of augmentations to add

Add another dataset using a previously generated `vocab.csv`:

```bash
./preprocesor/add_dataset.sh -f dataset1.csv -t tag1 -n 10 -m 196 -o true
```

- `-f` : PubChem bioassay .csv file, multiple can be specified
- `-t` : String specifying a tag to add to the preprocessed dataset, amount of `-t` and `-f` arguments must be the same
- `-n` : Positive integer representing amount of augmentations to add
- `-m` : Maximum length of tokens, any samples found that are longer are removed from the dataset
- `-o` : Boolean representing whether to ignore or override tokens not found in initial vocabulary

Train QSAR model using

```bash
./predictor/train_keras_rnn.py X.npy Y.npy 100 name
```

- First argument: A numpy file containing the tokenized X features
- Second argument: Must be labels in a numpy file
- Third argument: Amount of epochs to train for (positive integer), final argument specifies tag to name the model and training history

(Optional) Optimize accuracy post training with additional augmentations

```bash
./predictor/optimize_n.sh -x testX.npy -y testY.npy -m rnn_model.h5 -s 10
```

- `-x` : Numpy file containing tokenized X features
- `-y` : Numpy file containing labels
- `-m` : Path of model to optimize
- `-s` : (Optional) positive integer greater than 0, program will sample 1 in `-s` entries in `-x` and `-y` to evaluate

Generate chemicals

```bash
./inverse_qsar/inverse_qsar_cli.py args.json drugs_file.csv
```

- First argument : A JSON file containing all arguments to be used while generating
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
  - `augment` : Whether to augment data for model scoring function and how many times to augment
  - `max_len` : Maximum token length of molecules
  - `max_score` : Score to stop at
  - `prune_population` : Trim size of population
  - `target`: Target value for scoring functions to optimize
  - `file_name` : Pre-existing file of molecules to draw initial population from
- Second argument : A .csv to dump the molecules into  
- Third argument : Optional .csv file to dump fitness scores after training

## Scoring Functions

- `model` : Uses the QSAR model as a scoring function
- `lipinski` : Uses Lipinski's Rule of Five
- `qed` : Uses QED drug-likeness measure
- `pains_filter` : Filters out PAINS
- `custom_lipinski` : Uses a custom weighted version of Lipinski's Rule of Five

## Todo

- Add Flux model integration

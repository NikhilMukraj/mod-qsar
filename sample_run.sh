#!/bin/bash


cd preprocessor

python3 chembl_dataset_generator.py dataset_args.json -a aggregate.json -f true -n 10

cd ../predictor

python3 train_keras_rnn.py ../preprocessor/dopa_encoded_seqs.npy ../preprocessor/dopa_activity.npy
python3 train_keras_rnn.py ../preprocessor/sero_encoded_seqs.npy ../preprocessor/sero_activity.npy

PKG_OK=$(dpkg-query -W --showformat='${Status}\n' jq | grep "install ok installed")

if [[ -z "$PKG_OK" ]]
then
    sudo apt-get install jq
fi

tmp=$(mktemp)
fileout=`./optimize_n.sh -x ../preprocessor/sero_encoded_seqs.npy -y ../preprocessor/sero_activity.npy -m sero_rnn_model.h5 -s 100 | sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g'`
num="${fileout##*$'\n'}"

cd ../inverse_qsar

mv ../predictor/dopa_rnn_model.h5 dopa_rnn_model.h5
mv ../predictor/sero_rnn_model.h5 sero_rnn_model.h5

jq -n --arg num "$num" '.augment = [true, $num]' args.json > "$tmp" && mv "$tmp" args.json
python3 inverse_qsar_cli.py args.json generated_drugs/mols.csv generated_drugs/mols_scores.csv

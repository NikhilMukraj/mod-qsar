#!/bin/bash


cd preprocessor

curl -L -O dopamine.csv "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?task=resultdefs&aid=652054&start=1&limit=10000000&download=true&downloadfilename=AID_652054_pcget_bioassay_resultdefs&infmt=json&outfmt=csv"
curl -L -O serotonin.csv "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&aid=624169&version=1.1&response_type=save"

./preprocessor.sh -f dopamine.csv -f serotonin.csv -t dopa -t sero -n 10

cd ../predictor

python3 train_keras_rnn.py ../preprocessor/dopa_encoded_seqs.npy ../preprocessor/dopa_activity.npy
python3 train_keras_rnn.py ../preprocessor/sero_encoded_seqs.npy ../preprocessor/sero_activity.npy

PKG_OK=$(dpkg-query -W --showformat='${Status}\n' jq | grep "install ok installed")

if [[ -z "$PKG_OK" ]]
then
    sudo apt-get install jq
fi

tmp=$(mktemp)
fileout=`./optimize_n.sh | sed 's/\x1B\[[0-9;]\{1,\}[A-Za-z]//g'`
num="${fileout##*$'\n'}"

cd ../inverse_qsar

mv ../predictor/dopa_rnn_model.h5 dopa_rnn_model.h5
mv ../predictor/sero_rnn_model.h5 sero_rnn_model.h5

jq -n --arg num "$num" '.augment = [true, $num]' args.json > "$tmp" && mv "$tmp" args.json
python3 inverse_qsar_cli.py args.json generated_drugs/mols.csv generated_drugs/mols_scores.csv

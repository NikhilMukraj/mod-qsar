curl -L -O dopamine.csv link
curl -L -O serotonin.csv link

./preprocessor.sh -f dopamine.csv -f serotonin.csv -t dopa -t sero -n 10
python3 train_keras_rnn.py
python3 train_keras_rnn.py
./optimize_n.sh > file.txt

python3 inverse_qsar_cli.py args.json file

# notes

find way to get probability that molecule activates certain receptor
use that probability as reward

chembl sort by nervous system or drug classification (psycholeptics?)

search by filtered drug type and then find compound report card which has drug mechanisms?
or maybe download all drugs and then look through each

df = pd.read_csv(os.getcwd() + '\\bbbp_smi\\BBBP.csv', encoding='latin-1')

[https://pubchem.ncbi.nlm.nih.gov/bioassay/652054#section=Data-Table]
use CID to return smiles
[https://pubchem.ncbi.nlm.nih.gov/compound/43795#section=Canonical-SMILES&fullscreen=true]
scrape to get string

find smiles checker to ensure quality data

## genetic algo

represent tokens as binary, to mutate singular token flip random bit
if binary representation has a max value less than a power of two, add x amount to overflow back to zero if greater than that max
use enumeration of smiles strings sometimes to introduce more population/mutations/variability

## todo

- [ ] git submodule smiles enumeration script
- [X] preprocess smiles strings
- [ ] build and train rnn model
- [ ] add augmentation
- [ ] make figures from training
- [ ] modify existing genetic algo to have variable length chromosomes (variable n_bit length, set amount is equal to a token, decode into smiles string, if not a power of 2 add x amount to overflow back to 0)
- [ ] logarithmic scaling for reward based on activity prediction
- [ ] implement without checking for similarity yet, just initialize with random valid strings, penalize for being invalid
- [ ] add check for tanimoto similarity if smiles string is valid
- [ ] implementation of levenshtein distance in c to compare similarity of strings
- [ ] make figures

## useful links

[https://emoryraphael.medium.com/basic-sentiment-analysis-with-julia-using-lstm-e12d4754ee6b]
[https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services]
[https://www.kaggle.com/code/isaienkov/mechanisms-of-action-moa-prediction-eda]
[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6770028/]
[file:///C:/Users/nikhi/Downloads/ijms-20-04555-s001.pdf]
[https://www.kaggle.com/datasets/vladislavkisin/mlchem]
[http://cs230.stanford.edu/projects_winter_2020/reports/32618528.pdf]
[https://arxiv.org/pdf/1708.08227.pdf]

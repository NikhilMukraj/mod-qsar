# get name of file
# get tag to add to generated files
# get if debug

for i in "$@"
do
    if [[ $i == "-h" || $i == "-help" ]]
    then
        echo "
        -h : help command
        -f : filename containing raw bioassay .csv from pubchem
        -t : tag to add onto new preprocessed data files
        -n : must be a positive integer, optional, number of times to add augmented strings
        -m : max length of strings
        -o : true or false, optional, if new tokens are found they are removed from the dataset
        -d : true or false, optional, defaults to false, adds debug output
        "
        exit 0
    fi
done

while getopts f:t:d:n:m:o: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;
        t) tag=${OPTARG};;
        d) debug=${OPTARG};;
        n) num=${OPTARG};;
        m) max_len=${OPTARG};;
        o) override=${OPTARG};;
    esac
done

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

if [[ ! -f "vocab.csv" ]]
then
    printf "${RED}No existing vocab file${NC}\n"
    exit 1
fi

if [[ -z $filename || -z $tag ]]
then
    printf "${RED}Must have both an -f and -t argument${NC}\n"
    exit 1
fi

if [[ ! -f $filename ]]
then
    printf "${RED}${filename} is not found${NC}\n"
    exit 1
fi

if [[ -z $num ]]
then 
    num=10
fi

check_if_pos_int() {
    if [[ ! $1 =~ ^[0-9]+$ ]]
    then
        printf "${RED}${1} is not a positive integer${NC}\n"
        exit 1
    fi
}

check_if_pos_int $num

if [[ -z $max_len ]]
then 
    max_len="False"
else
    check_if_pos_int $max_len
fi

convert_to_bool() {
    result=$1
    if [[ -z $1 || $1 == "f" || $1 == "false" ]]
    then
        result="False"
    elif [[ $1 == "t" || $1 == "true" ]]
    then
        result="True"
    else
        echo "$1 is not true or false, defaulting to false"
        result="False"
    fi
}

debug="${debug,,}"
override="${override,,}"

convert_to_bool $debug
debug=$result
convert_to_bool $override
override=$result

if [[ ! -f "pkgs.so" ]]
then 
    echo "Julia sysimage not found, compiling PyCall sysimage..."
    julia -e 'using PackageCompiler; create_sysimage([:ProgressBars, :JLD, :PyCall], sysimage_path="pkgs.so")' || exit 1
fi

python3 initial_filter.py "$filename" $tag $override $debug || exit 1
printf "${GREEN}Finished filtering initial dataset${NC}\n"
julia --sysimage pkgs.so add_dataset.jl $tag $num $max_len $debug || exit 1
printf "${GREEN}Finished augmentations${NC}\n"
python3 final_preprocessing.py $tag $debug || exit 1

printf "${GREEN}Finished preprocessing strings${NC}\n"
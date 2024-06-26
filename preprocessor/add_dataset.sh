#!/bin/bash


for i in "$@"
do
    if [[ $i == "-h" || $i == "-help" ]]
    then
        echo "
        -h : help command
        -f : filename containing raw bioassay .csv
        -t : tag to add onto new preprocessed data files
        -n : must be a positive integer, optional, number of times to add augmented strings
        -m : max length of strings
        -o : true or false, if new tokens are found they are removed from the dataset
        -v : filename of vocabulary file, optional, defaults to vocab.csv
        -s : use sysimage, optional, defaults to false
        -d : true or false, optional, defaults to false, adds debug output
        "
        exit 0
    fi
done

while getopts f:t:d:n:v:s:m:o: flag
do
    case "${flag}" in
        f) filename=${OPTARG};;
        t) tag=${OPTARG};;
        n) num=${OPTARG};;
        m) max_len=${OPTARG};;
        o) override=${OPTARG};;
        v) vocab=${OPTARG};;
        s) sysimage=${OPTARG};;
        d) debug=${OPTARG};;
    esac
done

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

if [[ -z $vocab ]]
then
    vocab="vocab.csv"
fi

if [[ ! -f $vocab ]]
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

if [[ -z $vocab ]]
then
    vocab="vocab.csv"
elif [[ ! -z $vocab && ! -f $vocab ]]
then
    printf "${RED}$vocab not found${NC}"
    exit 1
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
convert_to_bool $sysimage
sysimage=$result
convert_to_bool $override
override=$result

if [[ $sysimage != "False" && ! -f "pkgs.so" ]]
then 
    echo "Julia sysimage not found, compiling PyCall sysimage..."
    julia -e 'using PackageCompiler; create_sysimage([:PyCall], sysimage_path="pkgs.so")' || exit 1
fi

python3 initial_filter.py "$filename" $tag $override $debug || exit 1
printf "${GREEN}Finished filtering initial dataset${NC}\n"

if [[ $sysimage != "False" ]]
then
    julia --sysimage pkgs.so add_dataset.jl $tag $num $max_len $override $vocab $debug || exit 1
else
    julia add_dataset.jl $tag $num $max_len $override $vocab $debug || exit 1
fi

printf "${GREEN}Finished augmentations${NC}\n"
python3 final_preprocessing.py $tag $debug || exit 1

printf "${GREEN}Finished preprocessing strings${NC}\n"

#!/bin/bash


for i in "$@"
do
    if [[ $i == "-h" || $i == "-help" ]]
    then
        echo "
        -h : help command
        -x : .npy file containing features
        -y : .npy file containing labels
        -m : .h5 keras model
        -v : optional, vocabulary file, assumes ../preprocessor/vocab.csv if not provided
        -s : optional, 1 in -s to sample from -x and -y, must be greater than 0
        -a : optional, integer minimum for range greater than 0, defaults to 2
        -b : optional, integer maximum for range greater than -a, defaults to 11
        -i : optional, integer increment or step, defaults to 2
        "
        exit 0
    fi
done

while getopts x:y:m:v:s:a:b:i flag
do
    case "${flag}" in
        x) x=${OPTARG};;
        y) y=${OPTARG};;
        m) model=${OPTARG};;
        s) sample=${OPTARG};;
        v) vocab=${OPTARG};;
        a) minimum=${OPTARG};;
        b) maximum=${OPTARG};;
        i) step=${OPTARG};;
    esac
done

RED='\033[0;31m'
GREEN='\033[1;32m'
NC='\033[0m'

if [[ -z $x || -z $y || -z $model ]]
then
    printf "${RED}Must have an -x, -y, and -m argument${NC}\n"
    exit 1
fi

if [[ ! -f $x ]]
then
    printf "${RED}$x not found\n"
    exit 1
elif [[ ! -f $y ]]
then
    printf "${RED}$y not found\n"
    exit 1
elif [[ ! -f $model ]]
then
    printf "${RED}$model not found\n"
    exit 1
fi

check_if_pos_int() {
    if [[ ! $1 =~ ^[0-9]+$ ]]
    then
        printf "${RED}${1} is not a positive integer${NC}\n"
        exit 1
    fi
}

if [[ ! -z $sample ]]
then
    check_if_pos_int $sample

    if [[ ! $1 == 0 ]]
    then
        printf "${RED}${sample} is 0${NC}\n"
        exit 1
    fi
else
    sample="False"
fi

if [[ ! -z $minimum ]]
then
    check_if_pos_int $minimum

    if [[ ! $1 == 0 ]]
    then
        printf "${RED}${minimum} is 0${NC}\n"
        exit 1
    fi
else
    minimum=2
fi

if [[ ! -z $maximum ]]
then
    check_if_pos_int $maximum

    if [[ ! $1 == 0 ]]
    then
        printf "${RED}${maximum} is 0${NC}\n"
        exit 1
    fi
else
    maximum=11
fi

if [[ ! -z $step ]]
then
    check_if_pos_int $step

    if [[ ! $1 == 0 ]]
    then
        printf "${RED}${step} is 0${NC}\n"
        exit 1
    fi
else
    step=2
fi

if [[ -z $vocab ]]
then
    vocab="../preprocessor/vocab.csv"
elif [[ ! -z $vocab && ! -f $vocab ]]
then
    printf "${RED}$vocab not found${NC}"
    exit 1
fi

if [[ ! -f "pkgs.so" ]]
then 
    printf "Julia sysimage not found, compiling Flux sysimage...\n"
    julia -e 'using PackageCompiler; create_sysimage([:Flux], sysimage_path="pkgs.so")' || exit 1
fi

python3 aug_accs_calc.py "$model" "$x" "$y" $sample "$vocab" $minimum $maximum $step || exit 1
printf "${GREEN}Finished calculating accuracy${NC}\n"
julia --sysimage pkgs.so optimize_n.jl || exit 1
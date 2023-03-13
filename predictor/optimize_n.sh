for i in "$@"
do
    if [[ $i == "-h" || $i == "-help" ]]
    then
        echo "
        -h : help command
        -x : .npy file containing features
        -y : .npy file containing labels
        -m : .h5 keras model
        -s : optional, 1 in -s to sample from -x and -y, must be greater than 0
        "
        exit 0
    fi
done

while getopts x:y:m:s: flag
do
    case "${flag}" in
        x) x=${OPTARG};;
        y) y=${OPTARG};;
        m) model=${OPTARG};;
        s) sample=${OPTARG};;
    esac
done

RED='\033[0;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
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

check_if_pos_int_and() {
    if [[ ! $1 =~ ^[0-9]+$ ]]
    then
        printf "${RED}${1} is not a positive integer${NC}\n"
        exit 1
    fi
}

if [[ -z $sample ]]
then
    check_if_pos_int $sample

    if [[ ! $1 =~ ^[0-9]+$ ]]
    then
        printf "${RED}${1} is 0${NC}\n"
        exit 1
    fi
else
    sample="False"
fi

if [[ ! -f "pkgs.so" ]]
then 
    printf "Julia sysimage not found, compiling Flux sysimage...\n"
    julia -e 'using PackageCompiler; create_sysimage([:Flux], sysimage_path="pkgs.so")' || exit 1
fi

python3 aug_accs_calc.py "$model" "$x" "$y" $sample || exit 1
printf "{GREEN}Finished calculating accuracy{NC}\n"
julia --sysimage pkgs.so optimize_n.jl || exit 1
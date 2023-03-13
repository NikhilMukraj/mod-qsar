for i in "$@"
do
    if [[ $i == "-h" || $i == "-help" ]]
    then
        echo "
        -h : help command
        -f : filename containing raw bioassay .csv from pubchem
        -t : tag to add onto new preprocessed data files
        -n : must be a positive integer, optional defaults to 5, number of times to add augmented strings
        -d : true or false, optional, defaults to false, adds debug output
        "
        exit 0
    fi
done

while getopts f:t:d:n: flag
do
    case "${flag}" in
        f) filenames+=("${OPTARG}");;
        t) tags+=("${OPTARG}");;
        d) debug=${OPTARG};;
        n) num=${OPTARG};;
    esac
done

RED='\033[0;31m'
GREEN='\033[1;32m'
CYAN='\033[1;36m'
NC='\033[0m'

if [[ -z $filenames || -z $tags ]]
then
    printf "${RED}Must have both an -f and -t argument${NC}\n"
    exit 1
fi

if [[ "${#filenames[@]}" != "${#tags[@]}" ]]
then
    printf "${RED}Must have same amount of -f and -t arguments${NC}\n"
    exit 1
fi

for i in "${filenames[@]}"
do
    if [[ ! -f $i ]]
    then
        printf "${RED}${i} is not found${NC}\n"
        exit 1
    fi
done

if [[ -z $num ]]
then 
    num=5
fi

check_if_pos_int() {
    if [[ ! $1 =~ ^[0-9]+$ ]]
    then
        printf "${RED}${1} is not a positive integer${NC}\n"
        exit 1
    fi
}

check_if_pos_int $num

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
convert_to_bool $debug
debug=$result

if [[ ! -f "pkgs.so" ]]
then 
    printf "Julia sysimage not found, compiling PyCall sysimage...\n"
    julia -e 'using PackageCompiler; create_sysimage([:JLD, :PyCall], sysimage_path="pkgs.so")' || exit 1
fi

for i in $(seq 0 $((${#filenames[@]}-1)))
do
    printf "Creating ${CYAN}${tags[$i]}${NC} dataset\n"
    python3 initial_filter.py "${filenames[$i]}" "${tags[$i]}" $debug || exit 1
done

printf "${GREEN}Finished filtering initial dataframes${NC}\n"

julia --sysimage pkgs.so generate_vocab.jl $num $debug ${tags[@]} || exit 1

printf "${GREEN}Finished augmentations${NC}\n"

for i in ${tags[@]}
do
    python3 final_preprocessing.py $i $debug || exit 1
done

printf "${GREEN}Finished preprocessing strings${NC}\n"
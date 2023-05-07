#!/bin/bash


pip install -r requirements.txt
julia install_pkgs.jl

cd smiles_tools
pip install .

cd ../c_wrapper
pip install .

GREEN='\033[1;32m'
NC='\033[0m'

printf "${GREEN}Finished initialization${NC}\n"

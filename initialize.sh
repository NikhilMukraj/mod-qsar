pip install -r requirements.txt
julia install_pkgs.jl

cd smiles_tools
pip install .

cd ../c_wrapper
pip install .

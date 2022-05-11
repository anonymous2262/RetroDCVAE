conda create -y -n retro python=3.6 tqdm
conda init
conda activate retro

# # CUDA 10.1
# conda install -y pytorch=1.6.0 torchvision cudatoolkit=10.1 torchtext -c pytorch

# CUDA 11.1  #https://www.zhaoyabo.com/?p=8291
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge


# Common dependencies
# install 'ipykernel' into the Python environment
conda install -n retro ipykernel --update-deps --force-reinstall

# pip dependencies
pip install gdown OpenNMT-py==1.2.0 networkx==2.5 selfies==1.0.3

# install rdkit
conda install -y rdkit -c conda-forge



# # install opennmt
# pip install OpenNMT-py==1.2.0
# # pip install xxx --ignore-installed

# # install networkx
# pip install networkx==2.5

# # install selfies
# pip install selfies==1.0.3

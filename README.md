# 02456 Molecular dynamics simulations using graph neural networks
This repository serves as a help to get you started with the project "Molecular dynamics simulations using graph neural networks" in 02456 Deep Learning. In particular, it provides an implementation of the [PaiNN](https://arxiv.org/pdf/2102.03150) model as well as a minimal example of how to train the PaiNN model on the QM9 dataset.

The repository should only be seen as a help and not a limitation in any way. You are free to modify and extend the code in any way you see fit or not use it all.

## Setup
To setup the code environment execute the following commands:
1. `git clone git@github.com:jonasvj/02456_painn_project.git`
2. `cd 02456_painn_project/`
3. `conda env create -f environment.yml`
4. `conda activate painn`
5. `pip install -e .`
6. `pip install torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html` (This might differ depending on your PyTorch / CUDA version - see https://github.com/rusty1s/pytorch_cluster?tab=readme-ov-file#installation)


## Provided modules
1. `src.data.QM9DataModule`: A PyTorch Lightning datamodule that you can use for loading and processing the QM9 dataset.
2. `src.models.PaiNN`: The PaiNN model implemented in PyTorch.
3. `src.models.AtomwisePostProcessing`: A module for performing post-processing of PaiNN's atomwise predictions. These steps can be slightly tricky/confusing but are necessary to achieve good performance (compared to normal standardization of target variables.)
4. `src.models.utils.EarlyStopping`: A simple class for doing early stopping.


## Usage
1. Run `python3 minimal_example.py`


## Next steps
1. Adapt the code (or make your own) to train PaiNN on the MD17 dataset or OMol25 dataset.
2. Use and evaluate your model for molecular dynamics simulations.
3. Do one of the two possible project directions.
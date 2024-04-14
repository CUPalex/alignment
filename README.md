# :hatched_chick: Dynamics of Human-LLM Alignment

## :egg: About
This project aims at shedding light at the dynamics of the development of the alignment between the word representations in LLMs and in the human brain.
In this project we used:
- [160M-parameter model from Pythia family](https://huggingface.co/EleutherAI/pythia-160m) as the LLM
- [Harry Potter reading dataset](http://www.cs.cmu.edu/~fmri/plosone/) as the human brain data. We used the already preprocessed version available on [gdrive](https://drive.google.com/drive/folders/1Q6zVCAJtKuLOh-zWpkS3lH8LBvHcEOE8).

The results of the work are presented in the ```dynamics_of_human_llm_alignment.pdf``` file in the root of the repository.

## :egg: Installation

Installation is available only by cloning the repository
```
git clone https://github.com/CUPalex/alignment.git
```

The ```environment.yml``` file contains the packages needed to run the code. To create the conda evironment from this file, run
```
conda env create -f environment.yml
```

## :egg: Usage

To compute the brain alignment scores for initial model representations, refer to ```run_all.py``` file.

To compute the alignment with residual representations with removed linguistic features, refer to ```feature_removal.py``` file.

To download the data from gdrive, refer to ```data/download_data.py``` file.


## :egg: Files structure

```
.
├── brain_alignment
│   └── computation.py # used to compute the alignment between model representations and brain representations
├── data
│   ├── data.py # used to get and align model data and brain data
│   └── download_data.py # script to download brain data from disk
├── model_representations
│   ├── feature_remover.py # used to remove linguistic features from model representations
│   ├── linguistic_features.py # used to calculate linguistic features for words
│   └── model_representations.py # used to get model representations for words
├── feature_removal.py
├── run_all.py
├── dynamics_of_human_llm_alignment.pdf
├── environment.yml
└── README.md
```

## :egg: Licence
MIT

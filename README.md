# Implications of shared motor and perceptual activations on the sensorimotor cortex for neuroprosthetic decoding

Code to recreate the figures from Silva*, Liu*, ..., and Chang. "Implications of shared motor and perceptual activations on the sensorimotor cortex for neuroprosthetic decoding", _Journal of Neural Engineering_, 2025. [10.1088/1741-2552/adf50e](https://doi.org/10.1088/1741-2552/adf50e).

## Installing the Python environment
An environment with all relevant dependencies can be installed using 
Anaconda. In a terminal, use the following command:
```bash
conda env create -f environment.yml
```
Once created, the environment must be activated to properly run the figure 
code. Activate the environment with the following command:
```bash
conda activate distractors
```

## Downloading relevant data files
Datasets needed to generate the figures can be accessed and downloaded via 
Zenodo [10.5281/zenodo.14011043](https://doi.org/10.5281/zenodo.14011043).

Once downloaded, place the files in a folder titled `data` within this 
repository. This repository should now have the following directory 
organization:
```
SilvaLiu2025_Distractors
└───data
└───distractors_figures
```
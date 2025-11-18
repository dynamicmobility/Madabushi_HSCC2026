# Towards Efficient Regions of Attraction Estimation for Hybrid Limit Cycles with Application to Legged Robots
This code repo contains the implementation of the reachable set algorithms described in the paper, used to compute
a forward-invariant tube for the simple bipedal walker model.

## Installation
### System Dependencies
The following instructions install the system dependencies to run the project on an Ubuntu 22.04 system.

This library depends on the `cdd` system library, which must first be installed with the command 
```
apt-get install -y libcdd-dev libgmp-dev
```
Further steps require a working `conda` install, which can be obtained [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).

### Python Dependencies
To install this python project and its dependencies, download the code repository and unzip.
From the root directory of the repo, run `bash install_env.sh`. 
This will create a conda environment named `immrax` with all required dependencies installed.

## Running the Code
All functional code used to generate the figures in the paper is contained within `Ellipsoids_Transformed.ipynb`.
Open the code notebook (It is recommended to use VSCode with the Jupyter extension) and select `immrax` as the kernel to use.
Finally, click "Run All."
The notebook will run, and generate all plots, which can be viewed interactively.
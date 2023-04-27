# ARGO Utils Package

This package contains functions that are used across multiple sub-packages within ARGO.

---

## Installation

### 1. Clone argo repo

First, clone the `argo` repository using SSH:

`$ cd <path-for-argo-repo>`

`$ git clone git@github.paypal.com:Simility-R/argo.git`

#### Example (clones argo repo to your home directory)

`$ cd ~`

`$ git clone git@github.paypal.com:Simility-R/argo.git`

Here the `argo` repo will be cloned to your home directory.

### 2. Create virtual environment

Now set up a virtual environment so that we can install the necessary dependencies for running the *ARGO Utils* sub-package:

#### 2a. Using venv

`$ cd <path-for-argo-utils-virtual-environment>`

`$ python3.7 -m venv argo_utils_venv`

##### Example (creates the virtual environment in your home directory)

`$ cd ~`

`$ python3.7 -m venv argo_utils_venv`

Here the argo virtual environment will be created in your home directory.

#### 2b. Using a conda environment

`$ conda create --name argo_utils_venv python=3.7`

### 3. Activate virtual environment

Now activate the environment so the necessary dependencies can be installed to that environment. The process for doing this depends on whether you've created the environment using ***venv*** or ***conda***.

#### 3a. Using venv

`$ source <path-to-argo-utils-virtual-environment>/bin/activate`

##### Example (assumes argo_utils_venv was created in home directory)

`$ source ~/argo_utils_venv/bin/activate`

#### 3b. Using a conda environment

`$ source activate argo_utils_venv`

### 4. Install sub-package and remote dependencies

Once the `argo_utils_venv` virtual environment is activated, you can install the *ARGO Utils* sub-package along with its dependencies by using the `setup.py` file (located in the parent directory of the sub-package). Note that the command doesn't explicitly reference the `setup.py` file, but it finds and processes the file in the location provided:

`$ pip install <path-to-argo-repo>/argo/argo_utils/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/argo_utils/.`

### 5. Install local dependencies

The *ARGO Utils* sub-package requires the *ARGO Rule Optimisation* sub-package to run. We can install this using the following command:

`$ pip install <path-to-argo-repo>/argo/rule_optimisation/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/rule_optimisation/.`

### 6. Install Jupyter

If you want to use the package in Jupyter Notebook or Jupyter Lab, you'll first need to install one of these in the virtual environment. With the virtual environment activated, run one of the following commands:

`$ pip install notebook` *(for Jupyter Notebook)*

`$ pip install jupyterlab` *(for Jupyter Lab)*

You'll then need to install the virtual environment itself as a Jupyter kernel. With the virtual environment activated, run the commands:

`$ pip install ipykernel`

`$ python -m ipykernel install --user --name argo_utils`

Select this kernel (`argo_utils`) when running the package in Jupyter Notebook/Lab.

## Queries or suggestions?

If you have any queries or suggestions please put them in the #sim-datatools-help Slack channel or email James directly.

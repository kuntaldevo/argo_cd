# ARGO Rule Optimisation Package

Contains modules related to rule optimisation:

1) **optimisation_functions** : Contains classes which are used to calculate optimisation metrics (e.g. Precision, Recall, FScore, etc). These are optimised for runtime vs the same functions from Sklearn.
2) **rule_optimiser** : Optimises a set of rules, given a labelled dataset.

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

Now set up a virtual environment so that we can install the necessary dependencies for running the *ARGO Rule Optimisation* sub-package:

#### 2a. Using venv

`$ cd <path-for-rule-optimisation-virtual-environment>`

`$ python3.7 -m venv rule_optimisation_venv`

##### Example (creates the virtual environment in your home directory)

`$ cd ~`

`$ python3.7 -m venv rule_optimisation_venv`

Here the argo virtual environment will be created in your home directory.

#### 2b. Using a conda environment

`$ conda create --name rule_optimisation_venv python=3.7`

### 3. Activate virtual environment

Now activate the environment so the necessary dependencies can be installed to that environment. The process for doing this depends on whether you've created the environment using ***venv*** or ***conda***.

#### 3a. Using venv

`$ source <path-to-rule-optimisation-virtual-environment>/bin/activate`

##### Example (assumes rule_optimisation_venv was created in home directory)

`$ source ~/rule_optimisation_venv/bin/activate`

#### 3b. Using a conda environment

`$ source activate rule_optimisation_venv`

### 4. Install sub-package and remote dependencies

Once the `rule_optimisation_venv` virtual environment is activated, you can install the *ARGO Rule Optimisation* sub-package along with its dependencies by using the `setup.py` file (located in the parent directory of the sub-package). Note that the command doesn't explicitly reference the `setup.py` file, but it finds and processes the file in the location provided:

`$ pip install <path-to-argo-repo>/argo/rule_optimisation/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/rule_optimisation/.`

### 5. Install local dependencies

The *ARGO Rule Optimisation* sub-package requires the *ARGO Rule Application*, *ARGO Utils* and *ARGO Rules* sub-packages to run. We can install these using the following commands:

`$ pip install <path-to-argo-repo>/argo/rule_application/.`

`$ pip install <path-to-argo-repo>/argo/argo_utils/.`

`$ pip install <path-to-argo-repo>/argo/rules/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/rule_application/.`

`$ pip install ~/argo/argo/argo_utils/.`

`$ pip install ~/argo/argo/rules/.`

### 6. Install Jupyter

If you want to use the package in Jupyter Notebook or Jupyter Lab, you'll first need to install one of these in the virtual environment. With the virtual environment activated, run one of the following commands:

`$ pip install notebook` *(for Jupyter Notebook)*

`$ pip install jupyterlab` *(for Jupyter Lab)*

You'll then need to install the virtual environment itself as a Jupyter kernel. With the virtual environment activated, run the commands:

`$ pip install ipykernel`

`$ python -m ipykernel install --user --name rule_optimisation`

Select this kernel (`rule_optimisation`) when running the package in Jupyter Notebook/Lab.

## Using the ARGO Rule Optimisation sub-package

The best examples of how to use the sub-package are the notebooks located in the *examples* folder. If you start a Jupyter Notebook/Lab window and navigate to that folder, you can open and run these example notebooks.

## Queries or suggestions?

If you have any queries or suggestions please put them in the #sim-datatools-help Slack channel or email James directly.

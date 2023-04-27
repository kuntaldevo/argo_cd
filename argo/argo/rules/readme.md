# ARGO Rules Package

Defines each rule in set as one (or all) of the following representations: string, dictionary, lambda expression or system-ready. One of these formats must be provided to define the rule set. The rules can then be reformatted into one of the other formats. There are multiple modules:

1) **rules** : Main class for defining a rule set. Utilises the three modules below to convert between formats.
2) **convert_rule_dicts_to_rule_strings** : Converts rules stored in the standard ARGO dictionary format into the standard ARGO string format.
3) **convert_rule_strings_to_rule_dicts** : Converts rules stored in the standard ARGO string format into the standard ARGO dictionary format.
4) **convert_rule_dicts_to_rule_lambdas** : Converts rules stored in the standard ARGO dictionary format into the standard ARGO lambda expression format.
5) **convert_rule_dicts_to_system_dicts** : Converts rules stored in the standard ARGO dictionary format into the system-ready format.
6) **convert_system_dicts_to_rule_dicts** : Converts rules stored in the system-ready format into the standard ARGO dictionary format.
7) **convert_rule_lambdas_to_rule_strings** : Converts rules stored in the standard ARGO lambda expression format into the standard ARGO string format.
8) **convert_processed_conditions_to_general**: Converts rules which contain either imputed or OHE features into the general format (i.e. so the rules can be applied to unprocessed data).

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

Now set up a virtual environment so that we can install the necessary dependencies for running the *ARGO Rules* sub-package:

#### 2a. Using venv

`$ cd <path-for-rules-virtual-environment>`

`$ python3.7 -m venv rules_venv`

##### Example (creates the virtual environment in your home directory)

`$ cd ~`

`$ python3.7 -m venv rules_venv`

Here the argo virtual environment will be created in your home directory.

#### 2b. Using a conda environment

`$ conda create --name rules_venv python=3.7`

### 3. Activate virtual environment

Now activate the environment so the necessary dependencies can be installed to that environment. The process for doing this depends on whether you've created the environment using ***venv*** or ***conda***.

#### 3a. Using venv

`$ source <path-to-rules-virtual-environment>/bin/activate`

##### Example (assumes rules_venv was created in home directory)

`$ source ~/rules_venv/bin/activate`

#### 3b. Using a conda environment

`$ source activate rules_venv`

### 4. Install sub-package and remote dependencies

Once the `rules_venv` virtual environment is activated, you can install the *ARGO Rules* sub-package along with its dependencies by using the `setup.py` file (located in the parent directory of the sub-package). Note that the command doesn't explicitly reference the `setup.py` file, but it finds and processes the file in the location provided:

`$ pip install <path-to-argo-repo>/argo/rules/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/rules/.`

### 5. Install Jupyter

If you want to use the package in Jupyter Notebook or Jupyter Lab, you'll first need to install one of these in the virtual environment. With the virtual environment activated, run one of the following commands:

`$ pip install notebook` *(for Jupyter Notebook)*

`$ pip install jupyterlab` *(for Jupyter Lab)*

You'll then need to install the virtual environment itself as a Jupyter kernel. With the virtual environment activated, run the commands:

`$ pip install ipykernel`

`$ python -m ipykernel install --user --name rules`

Select this kernel (`rules`) when running the package in Jupyter Notebook/Lab.

## Using the ARGO Rules sub-package

The best examples of how to use the sub-package are the notebooks located in the *examples* folder. If you start a Jupyter Notebook/Lab window and navigate to that folder, you can open and run these example notebooks.

## Queries or suggestions?

If you have any queries or suggestions please put them in the #sim-datatools-help Slack channel or email James directly.

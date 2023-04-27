# ARGO Simility APIs Package

Contains multiple modules for making API calls to a Simility environment:

1) **create_rule_in_simility** : API for creating a rule in a Simility instance, given the rule JSON config.
2) **update_rule_in_simility** : Updates an existing rule in Simility, given its stringified rule JSON config.
3) **return_rule_info_from_simility** : API for returning the rule configuration JSONs related to the rules found in a Simility environment.
4) **return_cassandra_entity_info** : API for returning the Cassandra entity information associated with a keyspace.

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

Now set up a virtual environment so that we can install the necessary dependencies for running the *ARGO Simility APIs* sub-package:

#### 2a. Using venv

`$ cd <path-for-simility-apis-virtual-environment>`

`$ python3.7 -m venv simility_apis_venv`

##### Example (creates the virtual environment in your home directory)

`$ cd ~`

`$ python3.7 -m venv simility_apis_venv`

Here the argo virtual environment will be created in your home directory.

#### 2b. Using a conda environment

`$ conda create --name simility_apis_venv python=3.7`

### 3. Activate virtual environment

Now activate the environment so the necessary dependencies can be installed to that environment. The process for doing this depends on whether you've created the environment using ***venv*** or ***conda***.

#### 3a. Using venv

`$ source <path-to-simility-apis-virtual-environment>/bin/activate`

##### Example (assumes simility_apis_venv was created in home directory)

`$ source ~/simility_apis_venv/bin/activate`

#### 3b. Using a conda environment

`$ source activate simility_apis_venv`

### 4. Install sub-package and remote dependencies

Once the `simility_apis_venv` virtual environment is activated, you can install the *ARGO Simility APIs* sub-package along with its dependencies by using the `setup.py` file (located in the parent directory of the sub-package). Note that the command doesn't explicitly reference the `setup.py` file, but it finds and processes the file in the location provided:

`$ pip install <path-to-argo-repo>/argo/simility_apis/.`

#### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/argo/simility_apis/.`

### 5. Install Jupyter

If you want to use the package in Jupyter Notebook or Jupyter Lab, you'll first need to install one of these in the virtual environment. With the virtual environment activated, run one of the following commands:

`$ pip install notebook` *(for Jupyter Notebook)*

`$ pip install jupyterlab` *(for Jupyter Lab)*

You'll then need to install the virtual environment itself as a Jupyter kernel. With the virtual environment activated, run the commands:

`$ pip install ipykernel`

`$ python -m ipykernel install --user --name simility_apis`

Select this kernel (`simility_apis`) when running the package in Jupyter Notebook/Lab.

## Using the ARGO Simility APIs sub-package

The best examples of how to use the sub-package are the notebooks located in the *examples* folder. If you start a Jupyter Notebook/Lab window and navigate to that folder, you can open and run these example notebooks.

## Queries or suggestions?

If you have any queries or suggestions please put them in the #sim-datatools-help Slack channel or email James directly.

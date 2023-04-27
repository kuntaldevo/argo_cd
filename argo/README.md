<!-- ![title](argo_logo.png) -->
<p align="center">
  <img src=argo_logo.png />
</p>

# ARGO - Automated Rule Generation and Optimisation

## What is ARGO?

ARGO is a fast, flexible and modular Python package for:

* Generating new fraud-capture rules using a labelled dataset.
* Optimising existing rules using a labelled or unlabelled dataset.
* Combining rule sets and removing/filtering those which are unnecessary.
* Generating rule scores based on their performance.
* Generating or updating existing system-ready configurations so that they can be updated in a Simility environment.

It aims to help streamline the process for developing a final, deployment-ready rule set for a client. Each task within this process has been modularised into sub-packages, so that these can be installed and run independently.

## How can I get ARGO?

### Using the ARGO docker image

The easiest way to use ARGO is to spin up a docker container using the `argo:v2.0.0` image. The `argo:v2.0.0` image is already loaded in both lilac machines and is ready to use - just spin up a container using this image (see below).

There is also a tar'd version of the image at `/tmp/argo_images` on both lilac machines. This can be loaded into docker using the command `docker load -i <path to image tar file>` (note that docker will need to be installed first). Then a container can be started using this image (see below).

#### Spin up the container

To spin up a container using the `argo:v2.0.0` image, use the following command:

`docker run -d --memory=<memory in GB> --cpus=<number of cores> -v <directory to mount>:/workdir/mount --name <container name> -p <machine port>:8080 argo:v2.0.0`

where:

* `<memory in GB>` is the amount of memory assigned to the container.
* `<number of cores>` is the number of cores assigned to the container.
* `<directory to mount>` is the path on the machine that you want to mount to the path `/workdir/mount` in the container. You have to mount the path(s) that need to be accessed in the container, otherwise you won't be able to access the required machine paths in the container.
* `<container name>` is the name for the container (this is just used to identify different running containers).
* `<machine port>` is the port on the machine that you want to map to the port in the container which is running jupyter lab. If running on on (eu)-lilac, you'll need to tunnel to this port before opening `localhost:<machine port>` in your browser to display the jupyter lab session.

##### Example

Say you want to spin up a container (named 'argo-james') on eu-lilac using the `argo:v2.0.0` image with 16GB of RAM and 6 cores. You want to mount your home directory to the container so that you can access the files in your home directory from the container. You wish to use port 1999 on eu-lilac and map that to the port in the container which is running jupyter lab.

First, SSH to eu-lilac. Then, in the terminal window (connected to eu-lilac), run the following command:

`docker run -d --memory=16GB --cpus=6 -v $HOME:/workdir/mount --name "argo-james" -p 1999:8080 argo:v2.0.0`

Once the command has ran successfully, tunnel to eu-lilac's port 1999 on your local machine, then open `localhost:1999` in your browser. The jupyter lab session should be displayed. In the left hand pane, there are two folders:

* *examples*: contains the example notebooks for running ARGO. The notebooks in the top level of the folder show how ARGO can be fully applied to a use case; the notebooks in the `subpackges_examples` folder give details on how each subpackage can be applied. **NOTE: if using one of the example notebooks on your own data, make sure you copy the notebook to your *mount* folder in the container. This will save the notebook to the machine (rather than in the container - the *example* folder will be deleted when the container is killed).**
* *mount*: this aligns to the machine path specified in the `docker run` command above. This was set as your home directory (using the environment variable `$HOME`), which means you can access your home directory from this folder in the container.

### Manual installation

If you prefer to install ARGO manually, follow the steps below.

#### Prerequsites

* Python 3.7 or above
* Access to Simility-R (<https://github.paypal.com/Simility-R>) - if you are unable to access the link, raise a request with Simility engineering team to be added to the DS group.
* Able to connect to the PayPal github (to clone the `argo` repository) - if you haven't set up the connection, follow the steps outlined here: <https://engineering.paypalcorp.com/confluence/display/git/Setting+up+Git+and+Github>

#### 1. Download the argo repository

You can download the `argo` repository using one of the following methods:

* Clone the repository using git
* Download the repository via github

##### 1a. Clone the repository using git

To clone the `argo` repository using SSH, use the following command:

`$ cd <path-for-argo-repo>`

`$ git clone git@github.paypal.com:Simility-R/argo.git`

###### Example (clones argo repo to your home directory)

`$ cd ~`

`$ git clone git@github.paypal.com:Simility-R/argo.git`

Here the `argo` repo will be cloned to your home directory.

##### 1b. Download the repository via github

To download the repo, go to <https://github.paypal.com/Simility-R/argo>, click the green `code` button in the top right hand corner and then click `Download ZIP`. This will download the repo as a ZIP folder, which can then be extracted.

#### 2. Create virtual environment

Now set up a virtual environment so that we can install the necessary dependencies for running the *ARGO* package:

##### 2a. Using venv

`$ cd <path-for-argo-virtual-environment>`

`$ python3.7 -m venv argo_venv`

###### Example (creates the virtual environment in your home directory)

`$ cd ~`

`$ python3.7 -m venv argo_venv`

Here the argo virtual environment will be created in your home directory.

##### 2b. Using a conda environment

`$ conda create --name argo_venv python=3.7`

#### 3. Activate virtual environment

Now activate the environment so the necessary dependencies can be installed to that environment. The process for doing this depends on whether you've created the environment using ***venv*** or ***conda***.

##### 3a. Using venv

`$ source <path-to-argo-virtual-environment>/bin/activate`

###### Example (assumes argo_venv was created in home directory)

`$ source ~/argo_venv/bin/activate`

##### 3b. Using a conda environment

`$ source activate argo_venv`

#### 4. Install ARGO and remote dependencies

*Note: If you encounter issues, try disconnecting from any VPNs*

Once the `argo_venv` virtual environment is activated, you can install the *ARGO* package along with its dependencies by using the `setup.py` file (located in the parent directory of the package). Note that the command doesn't explicitly reference the `setup.py` file, but it finds and processes the file in the location provided:

`$ pip install <path-to-argo-repo>/.`

##### Example (assumes argo repo is in home directory)

`$ pip install ~/argo/.`

#### 5. Install local dependencies

*Note: If you encounter issues, try disconnecting from any VPNs*

The *ARGO* package requires all of its sub-packages to be installed to run. We can install these by running the shell script found in the parent directory:

`$ sh <path-to-argo-repo>/install_subpackages.sh`

##### Example (assumes argo repo is in home directory)

`$ sh ~/argo/install_subpackages.sh`

#### 6. Install Jupyter

If you want to use the package in Jupyter Notebook or Jupyter Lab, you'll first need to install one of these in the virtual environment. With the virtual environment activated, run one of the following commands:

`$ pip install notebook` *(for Jupyter Notebook)*

`$ pip install jupyterlab` *(for Jupyter Lab)*

You'll then need to install the virtual environment itself as a Jupyter kernel. With the virtual environment activated, run the commands:

`$ pip install ipykernel`

`$ python -m ipykernel install --user --name argo`

Select this kernel (`argo`) when running the package in Jupyter Notebook/Lab.

## Using ARGO

The best examples of how ARGO can be fully applied to a use case are the notebooks located in the `examples` folder within the parent directory. There are also notebooks showing how to use each sub-package located in the `examples` folder in each sub-package's parent directory. Every notebook has a detailed explanation of what the package is aiming to accomplish, how to run it and how to interpret the outputs.

## Queries/suggestions

If you have any queries or suggestions please put them in the *#sim-datatools-help* channel or contact James directly.

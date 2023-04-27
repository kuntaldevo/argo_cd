# PARENT_DIRECTORY = $1
# MODULE_NAME = $1
cd argo
mkdir "$1"
mkdir "$1"/"$1"
mkdir "$1"/examples
mkdir "$1"/tests
touch "$1"/tests/__init__.py
touch "$1"/__init__.py
touch "$1"/"$1"/__init__.py
touch "$1"/setup.py
touch "$1"/readme.md
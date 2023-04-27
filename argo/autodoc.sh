rm -rf doc
mkdir doc
sphinx-apidoc -F -M -d 1 --separate -o doc argo argo/*setup*
cd doc
rm conf.py
cp ../argo_logo.png .
# Create conf.py file
cat >> conf.py <<EOL
import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.abspath('../argo/'))
project = 'ARGO'
copyright = '2021, Simility Data Team'
author = 'Simility Data Team'
html_logo = 'argo_logo.png'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx_rtd_theme',
]
autodoc_member_order = "bysource"
templates_path = ['_templates']
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
todo_include_todos = True
EOL
# Create style.css so page width is not limited
cat >> _static/style.css <<EOL
.wy-nav-content {
    max-width: none;
}
EOL
# Reference style.css in layout.html template
cat >> _templates/layout.html <<EOL
{% extends "!layout.html" %}
{% block extrahead %}
    <link href="{{ pathto("_static/style.css", True) }}" rel="stylesheet" type="text/css">
{% endblock %}
EOL
rm index.rst
# Generate index.rst (for index.html page)
cat >> index.rst <<EOL
ARGO: Automated Rule Generation and Optimisation
================================================

Introduction
------------

ARGO is a fast, flexible and modular Python package for:

* Generating new fraud-capture rules using a labelled dataset.
* Optimising existing rules using a labelled or unlabelled dataset.
* Combining rule sets and removing/filtering those which are unnecessary.
* Generating rule scores based on their performance.
* Generating or updating existing system-ready configurations so that they can be updated in a Simility environment.

It aims to help streamline the process for developing a final, deployment-ready rule set for a client. Each task within this process has been modularised into sub-packages, so that these can be installed and run independently.

Installation
------------

See https://github.paypal.com/Simility-R/argo for steps on installing ARGO.

Package documentation
---------------------

The below links contain the following information for each package:

* Example notebooks showing how the package can be used.
* Docstrings for the classes, methods or functions within the package.

.. toctree::
   :maxdepth: 3

   argo

EOL
# Add full ARGO examples (add relevant part to argo.rst)
cp ../examples/*ipynb .
sed -i '' "5i\\
Notebooks \\
--------- \\
\\
.. toctree:: \\
    \ \ \ \ \ \ :titlesonly: \\
    \\
    \ \ \ \ \ \ argo_example \\
    \ \ \ \ \ \ argo_unlabelled_example \\        
" argo.rst
sed -i '' "s/Module contents/ /" argo.rst
sed -i '' "s/---------------/  /" argo.rst
sed -i '' "s/\.\. automodule:: argo//" argo.rst
sed -i '' "s/:members:/ /" argo.rst
sed -i '' "s/:undoc-members:/ /" argo.rst
sed -i '' "s/:show-inheritance:/ /" argo.rst
echo ".. include:: ../READMEs/README_argo.rst" >> argo.rst

# Loop through sub-packages and add part for example notebooks
LIST_SUBPACKAGES="correlation_reduction read_data rule_application rule_filtering rule_generation rule_optimisation rule_scoring rules simility_apis simility_requests system_config_generation"
for subpackage in $LIST_SUBPACKAGES
do
    echo $subpackage    
    cp ../argo/$subpackage/examples/*.ipynb .    
    filename="argo.$subpackage.rst"        
    sed -i '' "5i\\
    Notebooks \\
    --------- \\
    \\
    .. toctree:: \\
        \ \ \ \ \ \ :titlesonly: \\
        \\
    " $filename
    for file in ../argo/$subpackage/examples/*.ipynb; do
        file_only="${file##*/}"
        file_no_extension="${file_only%%.*}"        
        sed -i '' "11i\\
            \ \ \ \ \ \ $file_no_extension \\
        " $filename
    done
    sed -i '' "s/Module contents/ /" $filename
    sed -i '' "s/---------------/  /" $filename
    sed -i '' "s/\.\. automodule:: argo.$subpackage.$subpackage.$subpackage//" $filename
    sed -i '' "s/:members:/ /" $filename
    sed -i '' "s/:undoc-members:/ /" $filename
    sed -i '' "s/:show-inheritance:/ /" $filename
    echo ".. include:: ../READMEs/README_$subpackage.rst" >> $filename
done
make clean
make html
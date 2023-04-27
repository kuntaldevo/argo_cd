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

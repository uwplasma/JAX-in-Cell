# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JAX-in-Cell'
copyright = '2025, UWPlasma'
author = 'UWPlasma'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
extensions = [
   'sphinx.ext.duration',
   'sphinx.ext.napoleon',
   'autoapi.extension'
]

autoapi_dirs = ['../jaxincell']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'piccolo_theme'

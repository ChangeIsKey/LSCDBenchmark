# Configuration file for the Sphinx documentation builder.
import sphinx_book_theme

import autoapi

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LSCDBenchmark'
copyright = '2023, Change is the key!'
author = 'Change is the key!'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
            "sphinx.ext.viewcode",
            "sphinx.ext.todo",
            "sphinx.ext.autodoc",
            "sphinx.ext.autosectionlabel",
            "sphinx.ext.napoleon",
            "sphinx.ext.intersphinx",
            # "sphinxcontrib.autodoc_pydantic",
            "autoapi.extension"
              ]

autoapi_dirs = ['../src']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    "matplotlib": ('https://matplotlib.org/stable/', None),
    "numpy": ('https://numpy.org/doc/stable/', None),
    "pandas": ('https://pandas.pydata.org/docs/', None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

autodoc_typehints_format = "fully-qualified"


autodoc_default_options = {
    'member-order': 'groupwise',
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': "BaseModel"
}
autodoc_typehints = 'signature'
autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
master_doc = 'index'
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_theme_path = [sphinx_book_theme.get_html_theme_path()]
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://github.com/ChangeIsKey/LSCDBenchmark",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
}

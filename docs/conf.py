# Configuration file for the Sphinx documentation builder.
import sphinx_book_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LSCDBenchmark'
copyright = '2023, Change is the key!'
author = 'Change is the key!'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['myst_parser', 'autoapi.extension']

autoapi_dirs = ['../src']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

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

# Set up Sphnix

Install Sphinx:

```sh
    pip install sphinx
```

Create a directory inside the project to hold the docs:

```sh
    cd /path/to/project
    mkdir docs
```

Run `sphinx-quickstart` to create the basic configuration in `docs`:

```sh
cd docs
sphinx-quickstart
```

Then, you will have `index.rst`, a `conf.py` and some other files. If you want to use Markdown instead, you have to firstly follow the installation steps in [Installation of Markdown](#markdown) and you can replace `index.rst` with `index.md`. Build the docs into html to see how they look:

```sh
    make html
```

The documentation `index.rst` (or `index.md` if you use Markdown) will be built as `index.html` in the output directory `_build/html/index.html`. You can check it by open `index.html` with any web browser.

<h2 id="markdown">Installation of Markdown</h2>

Both Markdown and reStructuredText can be used in the Sphnix project. You can install the Markdown setting by the following steps:

```sh
    pip install myst-parser
```

Add it in the `conf.py`:

```sh
    extensions = ["myst_parser"]
```

You need to identify the Table of Contents Tree in the `docs/index.md`. `toctree` tells that the other pages are the sub-page of the current page. The following `toctree` example is build upon the example sturcture.

```markdown
    ```{toctree}

    getting-started
    tasks/index
    developer
    glossary
    ```
```

The example structure of documents.

```markdown
myproject/
├── docs/
│   ├── index.md
│   ├── getting-started.md
│   ├── developer.md
│   ├── glossary.md
│   └── tasks/
│       ├── index.md
│       └── other pages...
```

The `toctree` about other pages in the `tasks` folder should be indicated in the `docs/tasks/index.md`.

```{seealso}
See [toctree](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html) for more infromation.
```

## Reference

- [Getting Started with Sphinx](https://docs.readthedocs.io/en/stable/intro/getting-started-with-sphinx.html#getting-started-with-sphinx)
- [Using Markdown with Sphnix](https://docs.readthedocs.io/en/stable/intro/)

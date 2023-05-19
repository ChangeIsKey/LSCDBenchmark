# Automatic API Documentation Generation

Install the `sphinx-autoapi`:

```sh
pip install sphinx-autoapi
```

Add the extension to `conf.py`:

```sh
extensions = ['autoapi.extension']
```

Set the configuration option about `autoapi_dirs`. It tells AutoAPI where to find the source content.

The Benchmark's source code was inside the `src` directory, which is similar to the following example.

```markdown
mypackage/
├── docs
│   ├── _build
│   ├── conf.py
│   ├── index.rst
│   ├── make.bat
│   ├── Makefile
│   ├── _static
│   └── _templates
├── README.md
└── src
    └── mypackage
        ├── _client.py
        ├── __init__.py
        └── _server.py#
```

In the `conf.py`, we then configure `autoapi_dirs` to be:

```sh
autoapi_dirs = ['../src']
```

The *Indices and tables* is included in the `index.rst` after you run `quick-start` while setting for sphnix. The reference link to page `genindex` in the `docs/index.rst`, you can see the content which is generated by AutoAPI. If you build the index page in markdown, you can create the *Indices and tables* with the following example.

```markdown
## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
```

The AutoAPI then will automatically generate the documentation and build a `toctree` list in the `doc/index.md`. If you don't want the list to be fully expand in the index page, you can set a `toctree` to contain the auto-generated page `autoapi/index.rst` with `:includehidden:`.

````markdown
```{toctree}
:includehidden:

autoapi/index.rst
```
````

````{note}
Autoapi generates several pages automatically. However, we might not be able to link all the pages in our index page. The error can happen as: ".../docs/autoapi/mypackage/src/index.rst: WARNING: document isn't included in any toctree".

You can set a `toctree` for the page in the index page. If you don't need it, you can hide it with `:hidden:` for the better layout of the page. It then won't show up in the index page.

```{toctree}
:maxdepth: 1
:hidden:

autoapi/mypackage/src/index.rst
```
````

See [this](https://sphinx-autoapi.readthedocs.io/en/latest/tutorials.html) for more information about autoapi tutorial.
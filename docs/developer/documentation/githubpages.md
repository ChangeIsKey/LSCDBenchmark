# Importing your docs

1. Connect [Read the Docs](https://readthedocs.org/) account to your GitHub account, which includes the repository of your project.

    ```{note}
    You need to have the ownership of the repository to be able to create the `ReadtheDocs pages.` If you are only one of the contributors of the repository, you can add the account of the owner to be one of the maintainers. ( `Admin` > `Maintainer` ) After the owner create the [webhook](https://docs.readthedocs.io/en/stable/glossary.html#term-webhook) in the setting page of the repository, the new Read the Docs page can be automatically built after pushing a new commit. 
    ```

2. Click on `Import a project` on the your account project dashbord. And, select the repository you want to import.

    - If there is an error for the missing `requirement.txt`, you can create one as `docs/requirements.txt` and include packages you have installed while you building the documentation, such as, `sphinx`, `myst-parser`, `sphinx-autoapi`, etc. Then, add the file path in the `.readthedocs.yaml` as following:

    ```sh
    python:
        install: 
            - requirements: docs/requirements.txt
    ```

3. Click on `View docs` to see how the page is built.

    ```{seealso}
    See [this](https://docs.readthedocs.io/en/stable/intro/import-guide.html#building-your-documentation) for more information about importing the documents.
    ```

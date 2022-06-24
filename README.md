# AI and Storytelling

## what?

This repository contains all the backstage information for the AI and
Storytelling course.

__Please note this is all subject to change__

## how?

Install [poetry](https://python-poetry.org/):

```sh
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

This snippet might be useful for activating the poetry environment without
having to create a new shell. Just type activate in a poetry directory and
it'll set you up:

```sh
alias activate="poetry env info --path && source \`poetry env info --path\`/bin/activate"
```

Install the project dependencies:

```sh
$ cd code
$ poetry install
```

Open up the `code` directory, and install the IPython kernel for the poetry
[environment](https://ka.ge/blog/2020/09/23/venv-jupyter.html):

```sh
$ poetry run python -m ipykernel install --user --name=aist
```

Then, you can start Jupyter in that directory:

```sh
$ jupyter lab
```

See `code/Demo.ipynb` for some examples.

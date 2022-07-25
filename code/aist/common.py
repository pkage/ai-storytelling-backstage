
def is_notebook() -> bool:
    '''
    Try to automatically detect if the code is running in an IPython
    notebook.

    :return: True if it's definitely an IPython notebook.
    '''
    try:
        # Attempt to resolve the ipython class name (type: ignore for linter)
        shell = get_ipython().__class__.__name__ # type: ignore

        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'Shell':
            # Colab
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False

    except NameError:
        # Probably standard Python interpreter
        return False

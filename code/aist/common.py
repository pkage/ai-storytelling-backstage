import torch
from IPython.display import Markdown, display
from typing import List

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

def is_gpu_available():
    '''
    Check if this environment supports gpu acceleration

    :return: True if acceleration is available
    '''
    return torch.cuda.is_available()


def render_output_text(output: List[str]):
    '''
    Attempt to import and render the passed text to an IPython shell via nice
    markdown rendering.

    Returns the original input (unchanged) if no notebook is detected.

    :param output: model output (text form)
    :return: either the original output or a transformation into an ipython notebook form
    '''

    if not is_notebook():
        return output

    text = ''
    for i, item in enumerate(output):
        item = item.replace('\n', '\n\n> ')
        text += f'Sample {i+1}:\n\n> {item}\n\n' 

    return display(Markdown(text))

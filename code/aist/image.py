import os
import warnings

from IPython.display import display
from min_dalle import MinDalle
import torch

from .common import is_notebook


# this is nasty but we have to make sure the HF model cache directory is
# created before we load in the transformers library
def _setup():
    # make sure that the models cache is set up
    if not os.path.exists('./model_cache/dalle'):
        os.makedirs('./model_cache/dalle')

    # also ignore warnings
    warnings.simplefilter('ignore')

_setup()


def _make_model(device=None, model_size='mini'):
    '''
    Create a dalle model.

    :param device: (optional) The device to initialize the model on. Set to None to autodetect.
    :param model_size: (optional) The size of model to use. Default 'mini'
    :return: An initialized DallE model.
    '''
    # automatically determine the device to use
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    is_mega = model_size == 'mega'

    model = MinDalle(
        models_root='./model_cache/dalle',
        dtype=torch.float32,
        device=device,
        is_mega=is_mega,
        is_reusable=True
    )

    return model


def generate_images(
        prompt,
        grid_size=2,
        model_size='mini',
        temperature=1,
        show_in_progress=False,
        accelerate=True,
        render=True,
        seed=None
    ):
    '''
    Generate a grid of images from a prompt.

    If used without rendering, returns a numpy array of images. These can be saved into pngs using Pillow:

    ```
    images = generate_images(..., render=False)
    image = Image.fromarray(images[i])
    image.save('image_{}.png'.format(i))
    ```

    :param prompt: The text prompt for the image generator.
    :param model_size: (optional) One of 'mini','mega'. Default 'mini'
    :param grid_size: (optional) The size of the grid. Default 3.
    :param temperature: (optional) How much variety to introduce into the outputs (higher means more variety). Default 1
    :param show_in_progress: (optional) Show in-progress images as they are rendered. Default False.
    :param accelerate: (optional) Whether to use GPU acceleration (if available). Default True
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: A grid of iamges matching the prompt.
    '''

    # min(dalle) expects the seed to be -1 as a default
    if seed is None:
        seed = -1

    # create the model
    model = _make_model(model_size=model_size, device=None if accelerate else 'cpu')

    if not render or not is_notebook():
        images = model.generate_images(
            text=prompt,
            seed=seed,
            grid_size=grid_size,
            is_seamless=False,
            temperature=temperature,
            top_k=256,
            supercondition_factor=16,
            is_verbose=False
        )

        return images.to('cpu').numpy()


    # determine if we're running in a notebook
    image_stream = model.generate_image_stream(
        text=prompt,
        seed=seed,
        grid_size=grid_size,
        progressive_outputs=show_in_progress,
        is_seamless=False,
        temperature=temperature,
        top_k=256,
        supercondition_factor=16,
        is_verbose=False
    )

    for image in image_stream:
        display(image)


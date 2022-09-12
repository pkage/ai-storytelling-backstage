import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import platform
import warnings

from IPython.display import display
from PIL import Image

# diffusion
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from .extras.safety import StableDiffusionSafetyCheckerDisable

# min(dalle)
from min_dalle import MinDalle
import requests

# captioning
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

# import torch with this flag to enable apple silicon fallback acceleration
import torch
from torch import Generator, autocast

from .common import is_notebook



HF_TOKEN_URL = 'https://cdn.ka.ge/vault/hf_aist.txt'
HF_TOKEN_LOCATION = os.path.expanduser('~/.huggingface')
HF_TOKEN_PATH = os.path.expanduser('~/.huggingface/token')

# this is nasty but we have to make sure the HF model cache directory is
# created before we load in the transformers library
def _setup():
    # make sure that the models cache is set up
    if not os.path.exists('./model_cache/dalle'):
        os.makedirs('./model_cache/dalle')

    # make sure that the huggingface directory exists when dropping a token
    if not os.path.exists(HF_TOKEN_LOCATION):
        os.makedirs(HF_TOKEN_LOCATION)

    # also ignore warnings
    if not 'HF_HOME' in os.environ:
        if not os.path.exists('./model_cache/hf-home'):
            os.makedirs('./model_cache/hf-home')

    warnings.simplefilter('ignore')

_setup()




def _get_device(device):
    '''
    Figure out which device we should be running stuff on.

    :param device: Device to prefer (None for auto)
    :return: 'cuda' or 'cpu' or 'mps'
    '''
    if device is None:
        if torch.cuda.is_available():
            return 'cuda'
        elif platform.machine() == 'arm64' and platform.system() == 'Darwin':
            # apple silicon acceleration
            return 'mps'
        else:
            return 'cpu'



def _make_dalle_model(device=None, model_size='mini'):
    '''
    Create a dalle model.

    :param device: (optional) The device to initialize the model on. Set to None to autodetect.
    :param model_size: (optional) The size of model to use. Default 'mini'
    :return: An initialized DallE model.
    '''
    # automatically determine the device to use

    device = _get_device(device)
    
    # patch: mps does not work for dalle
    if device == 'mps':
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


def _drop_hf_token():
    if not os.path.exists(HF_TOKEN_PATH):
        with open(HF_TOKEN_PATH, 'w') as token_file:
            token = requests.get(HF_TOKEN_URL).text

            token_file.write(token)


def _make_diffusion_model_text(device=None, unsafe=False):
    # simulates a login
    _drop_hf_token()

    # automatically determine the device to use
    device = _get_device(device)

    # pipeline kwargs
    kwargs = {
        'use_auth_token': True,
        'cache_dir': './model_cache/hf-home'
    }

    if device == 'cuda':
        kwargs['revision'] = 'fp16'
        kwargs['torch_dtype'] = torch.float16

    if unsafe:
        kwargs['safety_checker'] = StableDiffusionSafetyCheckerDisable

    pipe = StableDiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', 
        **kwargs
    )

    pipe = pipe.to(device)
    return pipe


def _make_diffusion_model_image(device=None, unsafe=False):
    # simulates a login
    _drop_hf_token()

    # automatically determine the device to use
    device = _get_device(device)

    # pipeline kwargs
    kwargs = {
        'use_auth_token': True,
        'cache_dir': './model_cache/hf-home'
    }

    if device == 'cuda':
        kwargs['revision'] = 'fp16'
        kwargs['torch_dtype'] = torch.float16

    if unsafe:
        kwargs['safety_checker'] = StableDiffusionSafetyCheckerDisable

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', 
        **kwargs
    )

    pipe = pipe.to(device)
    return pipe


def image_generation(
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
    model = _make_dalle_model(model_size=model_size, device=None if accelerate else 'cpu')

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


def stable_diffusion(prompt, accelerate=True, rounds=50, dims=(512,512), unsafe=False, seed=None):
    '''
    Generates an image from a text prompt.
    Powered by a stable diffusion pipeline.

    :param prompt: Text prompt to guide image generation
    :param rounds: (optional) How many inference rounds to do. More rounds yields more coherent results. Default 50
    :param dims: (optional) Dimensions of image to create. Default 512x512
    :param accelerate: (optional) Whether to use GPU acceleration (if available). Default True
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :return: 
    '''


    device = None if accelerate else 'cpu'

    model = _make_diffusion_model_text(device=device, unsafe=unsafe)

    device = _get_device(device)

    generator = None
    if seed is not None:
        if device == 'mps':
            device = 'cpu' # generators are not supported on MPS

        generator = Generator(device).manual_seed(seed)

    with autocast("cuda"):
        image = model(
            prompt,
            num_inference_steps=rounds,
            height=dims[1],
            width=dims[0],
            generator=generator
        )["sample"][0]

    return image


def stable_diffusion_img2img(image, prompt, dims=(512,512), rounds=50, strength=0.75, guidance_scale=7, unsafe=False, accelerate=True, seed=None):
    '''
    Generates an image from a source image, guided by a text prompt.
    Powered by a stable diffusion pipeline.

    :param image: Initial image to work on. Pass either a path or a Pillow image
    :param prompt: Text prompt to guide image generation
    :param rounds: (optional) How many inference rounds to do. More rounds yields more coherent results. Default 50
    :param dims: (optional) Dimensions to scale output image to. Default 512x512
    :param strength: (optional) How much noise to add to the image between 0 and 1 (lower=less noise). Low values correspond to outputs closer to the input. Default 0.75
    :param guidance_scale: (optional) How much to weight the text prompt. Default 7
    :param accelerate: (optional) Whether to use GPU acceleration (if available). Default True
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :return: 
    '''

    # if it's a string, assume a path and open the image
    if type(image) is str:
        image = Image.open(image)

    # convert colorspace and ensure sizing
    image = image.convert('RGB').resize(dims)

    device = None if accelerate else 'cpu'

    model = _make_diffusion_model_image(device=device, unsafe=unsafe)

    device = _get_device(device)

    generator = None
    if seed is not None:
        if device == 'mps':
            device = 'cpu' # generators are not supported on MPS

        generator = Generator(device).manual_seed(seed)


    with autocast("cuda"):
        image = model(
            prompt=prompt,
            init_image=image,
            num_inference_steps=rounds,
            strength=strength,
            guidance_scale=guidance_scale,
            generator=generator
        )["sample"][0]

    return image



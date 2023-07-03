# models

## Technical details

We can [run Hugging Face transformers
locally](https://huggingface.co/docs/transformers/installation) via the
`transformers` package on PyPI.

```python
from transformers import pipeline

print(pipeline('sentiment-analysis')('we love you'))
# >>> [{'label': 'POSITIVE', 'score': 0.9998704791069031}]
```

These can also run offline with some [mild
shenanigans](https://huggingface.co/docs/transformers/installation#offline-mode).

## Models

Here is a list of pipelines that we can (ab)use for our purposes. Note that
Hugging Face is really only built for transformer systems.

### Text → Text

There are broadly two categories here: generative continuation from a prompt,
and direct answering two questions from the user. There's a lot of overlap
between categories, as prompt continuation tends to answer questions as well.

Generates text from prompts:

_GPT-2 deriviatives:_

- [GPT2](https://huggingface.co/gpt2) from OpenAI
- [DistilGPT2](https://huggingface.co/distilgpt2) from Hugging Face (distilled from GPT2)
- [GPT2-Large](https://huggingface.co/gpt2-large) from OpenAI (⭐️ i like this one)

_Open Pre-trained Transformers (OPT) model family:_

- [OPT-350m](https://huggingface.co/facebook/opt-350m) from FB
- [OPT-1.3b](https://huggingface.co/facebook/opt-1.3b) from FB

[See more sizes here](https://huggingface.co/models?arxiv=arxiv:2205.01068)

Fills in masked tokens (`hello, i am a [MASK] model` → `hello, i am a role model`):

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

Summarization:

- [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)

Question answering:

- [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
- [tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2)


### Text → Image

Image generation from text prompt:

- [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) is the main one.
- [dalle-mini](https://huggingface.co/dalle-mini/dalle-mini) minimization of OpenAI's Dall-E model.

We mainly focus on stable diffusion. We also use stable diffusion to provide inpainting and image-to-image transformations.

### Image → Text

Captioning images:

- [ViT-GPT2 Image Captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning)

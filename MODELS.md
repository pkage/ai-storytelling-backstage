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
- [GPT-J-6B](https://huggingface.co/EleutherAI/gpt-j-6B) from EleutherAI (⭐️ i like this one)

_Open Pre-trained Transformers (OPT) model family:_

- [OPT-350m](https://huggingface.co/facebook/opt-350m) from FB
- [OPT-1.3b](https://huggingface.co/facebook/opt-1.3b) from FB

[See more sizes here](https://huggingface.co/models?arxiv=arxiv:2205.01068)

Fills in masked tokens:
(`hello, i am a [MASK] model` → `hello, i am a role model`):

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

Summarization:

- [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)

Translation:

- [m2m100_418M](https://huggingface.co/facebook/m2m100_418M) from FB
- [mt5](https://huggingface.co/google/mt5-base) from Google


Question answering:

- [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)
- [tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2)


### Text → Image

_it's tricky to find pre-existing models for this space as they're not tagged. we can always make our own_

Image generation from text prompt:

- [dalle-mini](https://huggingface.co/dalle-mini/dalle-mini) minimization of OpenAI's DallE. Big on the internet!
- [BigGAN-deep-128](https://huggingface.co/osanseviero/BigGAN-deep-128) only works with imagenet inputs

### Image → Text

_it's tricky to find pre-existing models for this space as they're not tagged. we can always make our own_

Optical character recognition:

- [manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base) OCR for Manga
- [trocr](https://huggingface.co/microsoft/trocr-base-printed) OCR from Microsoft

### Image → Image

_it's tricky to find pre-existing models for this space as they're not tagged. we can always make our own_

Style-transfer:

- [tf-neural-style-transfer](https://huggingface.co/Shamima/tf-neural-style-transfer)

### Recommender systems

- [recommender transformers](https://huggingface.co/keras-io/recommender-transformers)

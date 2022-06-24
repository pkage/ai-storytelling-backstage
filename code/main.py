#! /usr/bin/env python

from transformers import pipeline, set_seed

def _seed_if_necessary(seed):
    if seed is not None:
        set_seed(seed)

def summarization(
        text,
        model='facebook/bart-large-cnn',
        max_length=130,
        min_length=30,
        do_sample=False,
        seed=None
    ):

    _seed_if_necessary(seed)

    pipe = pipeline(task='summarization', model=model)

    return pipe(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample
    )


def text_generation(
        prompt, 
        max_length=200,
        num_return_sequences=3,
        model='small',
        seed=None
    ):

    _seed_if_necessary(seed)

    if model in ['small', 'medium', 'large']:
        mapping = {
            'small': 'distilgpt2',
            'medium': 'gpt2',
            'large': 'EleutherAI/gpt-j-6B'
        }
        model = mapping[model]

    pipe = pipeline(task='text-generation', model=model)

    return pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )



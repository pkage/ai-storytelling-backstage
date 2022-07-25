#! /usr/bin/env python

import os
from textwrap import dedent
from .common import is_notebook

# this is nasty but we have to make sure the HF model cache directory is
# created before we load in the transformers library
def _setup():
    # make sure that the models cache is set up
    if not 'TRANSFORMERS_CACHE' in os.environ:
        if not os.path.exists('./model_cache/hf'):
            os.makedirs('./model_cache/hf')

        # set the environment variable
        os.environ['TRANSFORMERS_CACHE'] = os.path.abspath('./model_cache/hf')

_setup()


from transformers import pipeline, set_seed # type: ignore
from transformers.utils.logging import set_verbosity_error

from typing import List, Any
from IPython.display import Markdown, display


# silence errors from deep within Transformers
set_verbosity_error()


def _seed_if_necessary(seed):
    '''
    Quick function that sets the seed if it's passed. If not provided,
    no seed will be explicitly set

    :param seed: Seed to set or None
    '''
    if seed is not None:
        set_seed(seed)


def _render_output_text(output: List[str]):
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


def summarization(
        text,
        model='facebook/bart-large-cnn',
        max_length=130,
        min_length=30,
        do_sample=False,
        seed=None,
        render=True
    ):
    '''
    Summarize text from a prompt.

    :param text: The text to summarize.
    :param model: (optional) The model to use for summarization (default `facebook/bart-large-cnn`)
    :param max_length: (optional) The minimum length of the summary. (default 30)
    :param min_length: (optional) The maximum length of the summary. (default 130)
    :param do_sample: (optional) Whether to subsample input (default False)
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: A summarization of the original text.
    '''
    _seed_if_necessary(seed)

    pipe = pipeline(task='summarization', model=model)

    results = pipe(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=do_sample
    )

    # convert results to a list of strings
    results = [r['summary_text'] for r in results]

    # skip render attempt if not requested
    if not render:
        return results

    return _render_output_text(results)


def text_generation(
        prompt, 
        max_length=200,
        num_return_sequences=3,
        model='small',
        seed=None,
        render=True
    ):
    '''
    Generate text from a prompt.

    Options for model are:
        - 'small', 'medium', 'large' (mapping to distilgpt2, gpt2, and gpt-j-6b)
        - any other text model on HF

    :param prompt: Text to prompt the pipeline with.
    :param model: (optional) Model to use. Default 'small'.
    :param max_length: (optional) Length of text to generate. Default 200.
    :param num_return_sequences: (optional) Number of different responses to make. Default 3.
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True

    :returns: A list of text generated by the model.
    '''

    _seed_if_necessary(seed)

    if model in ['small', 'medium', 'large']:
        mapping = {
            'small': 'distilgpt2',
            'medium': 'gpt2',
            'large': 'EleutherAI/gpt-j-6B'
        }
        model = mapping[model]

    pipe = pipeline(task='text-generation', model=model)

    results = pipe(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences
    )

    # convert results to a list of strings
    results = [r['generated_text'] for r in results]

    # skip render attempt if not requested
    if not render:
        return results

    return _render_output_text(results)



def sentiment_analysis(
        text,
        model='distilbert-base-uncased-finetuned-sst-2-english',
        seed=None,
        render=True
    ):
    '''
    Analyze the sentiment of a particular piece of text.

    :param text: The text to analyze.
    :param model: (optional) The model to use for analysis.
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: The most likely sentiment of the text.
    '''
    _seed_if_necessary(seed)

    pipe = pipeline(task='sentiment-analysis', model=model)

    sentiment = pipe(text)[0]

    if not render:
        return sentiment

    color = 'red' if sentiment['label'] == 'NEGATIVE' else 'green';
    return display(Markdown(f'<span style="color: {color};">{sentiment["label"]}</span> (score: {sentiment["score"]*100}%)'))


def fill_mask(
        text,
        model='bert-base-uncased',
        seed=None,
        render=True
    ):
    '''
    Guess words that fill a specific slot in some text. The default mask token is [MASK].

    :param text: The text to fill, with the mask token in it.
    :param model: (optional) The model to use.
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: A string with the mask filled in.
    '''
    _seed_if_necessary(seed)

    pipe = pipeline(task='fill-mask', model=model)

    masks = pipe(text)

    if not render:
        return masks

    # helper to render a string as bold
    def render_mask_result(i, obj):
        bolded = obj['sequence'].replace(obj['token_str'], f'**<span style="color: blue;">{obj["token_str"]}</span>**')

        return f'Sample {i+i} ({100*obj["score"]:4}\n\n> {bolded}'

    # render each item with a seqeunce number
    masks = '\n\n'.join([render_mask_result(i, obj) for i, obj in enumerate(masks)])

    return display(Markdown(masks))


def question_answering(
        question,
        context,
        model='deepset/roberta-base-squad2',
        seed=None,
        render=True
    ):
    '''
    Answer a question about some given context.

    :param question: The question to answer from the data.
    :param context: The context from which to draw the answer.
    :param model: (optional) The model to use.
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: 
    :rtype: 
    '''
    _seed_if_necessary(seed)

    pipe = pipeline(task='question-answering', model=model)

    answer = pipe({
        'question': question,
        'context': context
    })

    if not render:
        return answer

    return display(Markdown(dedent(f'''\
        Answer: **{answer['answer']}**

        Score: `{answer['score']}`  
        Position: {answer['start']} to {answer['end']}
    ''')))


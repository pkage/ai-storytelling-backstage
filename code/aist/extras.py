from IPython.display import Markdown, display
from .text import sentiment_analysis


def sentiment_ranking(strings, render=True, seed=None):
    """
    Rank a list of strings by sentiment

    :param strings: 
    :param seed: (optional) Seed value for reproducible pipeline runs.
    :param render: (optional) Automatically render results for an ipython notebook 
                   if one is detected. Default True
    :return: The strings ranked by sentiment.
    """
    results = []

    for string in strings:
        result = sentiment_analysis(string, seed=seed, render=False)
        result['original'] = string

        # make a 'sort score', which just makes sure that 
        # negative sentiments drop to the bottom.
        result['sort_score'] = result['score']
        if result['label'] == 'NEGATIVE':
            result['sort_score'] *= -1

        results.append(result)

    # sort in reverse so positive sentiments are at the top
    results = sorted(results, key=lambda x: x['sort_score'], reverse=True)

    if not render:
        return results

    # generate the markdown

    output_md = ''

    for i, result in enumerate(results):
        color = 'red' if result['label'] == 'NEGATIVE' else 'green';
        output_md += f'{i+1}. **{result["original"]}**\n\n    (<span style="color: {color};">{result["label"]}</span> score: {result["score"]*100:.5}%)\n\n'

    return display(Markdown(output_md))

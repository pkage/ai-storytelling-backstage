#! /usr/bin/env python

from transformers import pipeline, set_seed
from pprint import pprint

from aist import text
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# generator = pipeline(
#         task='text-generation',
#         model='distilgpt2',
#         tokenizer=tokenizer
# )

# results = generator(['My name is Buck and i like to '])


article = '''In a world far away, there was a time when a certain 
young man sat in a chair and wrote a story, one that 
would later be shared over a million different screens, 
when it would create change, when it would be seen. That 
young man wrote a story and called it Star Trek and it 
created a universe. It was written by a creator whose 
name is Leonard Nimoy. Leonard Nimoy has an unbelievable, 
and somewhat unfair, status in pop culture. He has an 
incredible and almost impossible to live up to level of 
fame. Even after the success of Star Trek he was typecast 
as Spock. That is what led to the creation of the 
character of Data on Star Trek: The Next Generation. He 
played a character who had a sense of logic and a sense 
of justice. To be able to come up with something as 
revolutionary as that as an actor is not an easy trick. 
In fact he does it in this interview. He did it without 
being typecast and did a hell'''

# generator = pipeline('text-generation', model='EleutherAI/gpt-j-6B')

# set_seed(420)

# results = generator("In a world far away", max_length=200, num_return_sequences=1)


results = text.summarization(article)
summary = results[0]['summary_text']

print(summary)
print(text.sentiment_analysis(summary))






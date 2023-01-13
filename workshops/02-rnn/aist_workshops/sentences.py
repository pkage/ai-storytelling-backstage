import numpy as np
import spacy

nlp = spacy.load('en_core_web_sm')

class SentenceEncoder:
    mapping = {}
    reverse_mapping = {
        0: ''
    }
    
    def __init__(self, sentence):
        doc = nlp(sentence)
        
        for i, token in enumerate(doc):
            print(f'{i+1} : {token.text} ({token.lemma_})')
            self.mapping[token.lemma_] = i+1
            self.reverse_mapping[i+1] = token.lemma_


    def lemmatize(self, sentence):
        doc = nlp(sentence)
        
        out = []
        for i, token in enumerate(doc):
            self.mapping[token.text.lower()] = i+1
            out.append(token.lemma_)

        return out


    def _encode_word(self, lemma, eol=False):
        encoding = np.zeros(len(self.mapping)+2)

        if eol:
            encoding[-1] = 1
            return encoding

        if lemma in self.mapping:
            i = self.mapping[lemma]
            encoding[i] = 1

        return encoding


    def _decode_word(self, vector):
        nonzero = (vector != 0).argmax()

        print(nonzero)
        print(len(vector))
        if nonzero+1 == len(vector):
            word = '(end)'
            return word
        word = self.reverse_mapping[nonzero]
        return word


    def encode(self, sentence):
        doc = nlp(sentence)

        out = []
        for token in doc:
            out.append(self._encode_word(token.lemma_))
        
        out.append(self._encode_word(None, eol=True))

        return out


    def decode(self, vectors):
        out = []
        for vector in vectors:
            word = self._decode_word(vector)
            out.append(word)
        return out
           
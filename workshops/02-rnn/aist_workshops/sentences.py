import numpy as np
import spacy

nlp = spacy.load('en_core_web_md')

class SentenceEncoder:
    mapping = {}
    reverse_mapping = {
        0: ''
    }
    
    def __init__(self, word_list):
        '''
        Construct the encoder with a pre-supplied word list

        
        '''
        doc = nlp(word_list)
        
        for i, token in enumerate(doc):
            print(f'{i+1} : {token.text} ({token.lemma_})')
            self.mapping[token.lemma_] = i+1
            self.reverse_mapping[i+1] = token.lemma_


    def lemmatize(self, word):
        doc = nlp(word)
        
        out = []
        for i, token in enumerate(doc):
            out.append(token.lemma_)

        return out


    def _encode_word(self, lemma, eol=False, similarity=True):
        encoding = np.zeros(len(self.mapping)+2)

        if eol:
            encoding[-1] = 1
            return encoding

        if not similarity:
            if lemma in self.mapping:
                i = self.mapping[lemma]
                encoding[i] = 1
        else:
            for i in range(1,len(encoding)-1):
                encoding[i] = nlp(self.reverse_mapping[i])[0].similarity(nlp(lemma)[0])

        return encoding


    def _decode_word(self, vector, similarity=True):
        if similarity:
            vector = (vector == 1)
        nonzero = vector.argmax()

        if nonzero+1 == len(vector):
            word = '(end)'
            return word
        word = self.reverse_mapping[nonzero]
        return word


    def encode(self, sentence, similarity=True, noisy=False):
        doc = nlp(sentence)
        out = []
        for token in doc:
            encoded = self._encode_word(token.lemma_, similarity=similarity)
            if noisy:
                print(f'{encoded} -> ({token.lemma_})')
            out.append(encoded)
        
        out.append(self._encode_word(None, eol=True))

        return out


    def show_encoded(self, encoded):
        for line in encoded:
            for el in line:
                if el == 1:
                    print('🟩', end='')
                else:
                    print('⬜️', end='')
            print(f' -> {self._decode_word(line)}')


    def decode(self, vectors, similarity=True):
        out = []
        for vector in vectors:
            word = self._decode_word(vector, similarity=similarity)
            out.append(word)
        return out
    

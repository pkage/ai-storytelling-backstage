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
        self.mapping = {}
        self.reverse_mapping = {
            0: ''
        }
        doc = nlp(word_list)
        
        for i, token in enumerate(doc):
            print(f'{i+1} : {token.text} ({token.lemma_})')
            self.mapping[token.lemma_] = i+1
            self.reverse_mapping[i+1] = token.lemma_


    def lemmatize(self, sentence):
        '''
        Tokenize and lemmatize a sentence

        :param sentence: A sentence
        :type sentence: str
        :return: Lemmas for every token
        :rtype: List[str]
        '''
        doc = nlp(sentence)
        
        out = []
        for i, token in enumerate(doc):
            out.append(token.lemma_)

        return out


    def _encode_word(self, lemma, eol=False, similarity=True):
        '''
        Encode a word with the encoding map

        :param lemma: A sentence
        :type lemma: str
        :param eol: whether or not this should be an EOL token
        :type eol: bool
        :param similarity: whether or not to use the spacy similarity
        :type similarity: bool
        :return: Encoding distribution
        :rtype: np.array(dtype=float)
        '''

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
        '''
        Decode a word with the encoding map

        :param vector: A sentence
        :type vector: np.array(dtype=float)
        :param similarity: whether or not to use the spacy similarity
        :type similarity: bool
        :return: word
        :rtype: str
        '''
        if similarity:
            vector = (vector == 1)
        nonzero = vector.argmax()

        if nonzero+1 == len(vector):
            word = '(end)'
            return word
        word = self.reverse_mapping[nonzero]
        return word



    def encode(self, sentence, similarity=False, noisy=False):
        '''
        Encode a sentence with the encoding map

        :param sentence: A sentence
        :type sentence: str
        :param similarity: whether or not to use the spacy similarity
        :type similarity: bool
        :param noisy: whether or not to print to the terminal
        :type noisy: bool
        :return: List of encoding distributions
        :rtype: List[np.array(dtype=float)]
        '''
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
        '''
        Show the encoding information as emoji

        :param encoded: Encoding distribution
        :type encoded: List[np.array(dtype=float)]
        '''
        for line in encoded:
            for el in line:
                if el == 1:
                    print('🟩', end='')
                else:
                    print('⬜️', end='')
            print(f' -> {self._decode_word(line)}')


    def decode(self, vectors, similarity=False):
        '''
        Decode a string with the decoding map

        :param vectors: List of encoding distributions
        :type vectors: List[np.array(dtype=float)]
        :param similarity: whether or not to use the spacy similarity
        :type similarity: bool
        :return: A decoded sentence
        :rtype: List[str]
        '''
        out = []
        for vector in vectors:
            word = self._decode_word(vector, similarity=similarity)
            out.append(word)
        return out
    

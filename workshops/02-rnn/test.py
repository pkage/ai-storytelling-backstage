from pprint import pprint
from aist_workshops.sentences import SentenceEncoder

s = SentenceEncoder('This is a test.')

encoded = s.encode('this test is not singularly a test.', similarity=True)
print('encoded:')
pprint(encoded)

print(s.lemmatize('this test is not a test.'))

decoded = s.decode(encoded)
print('decoded', decoded)



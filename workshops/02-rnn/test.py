from aist_workshops.sentences import SentenceEncoder

s = SentenceEncoder('This is a test.')

encoded = s.encode('this test is not a test.')
print('encoded', encoded)

print(s.lemmatize('this test is not a test.'))

decoded = s.decode(encoded)
print('decoded', decoded)



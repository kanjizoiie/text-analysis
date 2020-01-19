from nltk import bigrams, trigrams
from nltk.corpus import brown
from collections import Counter, defaultdict

model = defaultdict(lambda: defaultdict(lambda: 0))

# Count the frequency of the combined words
for sentence in brown.sents():
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1

# Convert the frequencies to probabilities
for w1_w2 in model:
    total = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total

# Print the frequencies for the model when using specific words
print('Example:', model["the", "price"])


while True:
    fw = input("Please input a string: ")
    sw = input("Please input a string: ")
    print('{}|{} - {}'.format(fw, sw, model[fw, sw]))

    
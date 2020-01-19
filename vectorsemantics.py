from nltk.corpus import reuters, brown
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial

model = defaultdict(lambda: defaultdict (lambda: 0))
context_size = 6

dataset = reuters.words()

index = 0
# Count the words and make record of the frequencies.
for index in range(len(dataset)):
    context = dataset[index - context_size:index + context_size]
    context = [c_word.lower() for c_word in context]
    for word in context:
        model[dataset[index].lower()][word] += 1
    index += 1

comp = ("texaco", "oil", "money")
comp1 = ("bank", "oil", "money")

x = model[comp[0]][comp[1]]
y = model[comp[0]][comp[2]]

x1 = model[comp1[0]][comp1[1]]
y1 = model[comp1[0]][comp1[2]]

print('The similarity is:', spatial.distance.cosine((x, y), (x1, y1)))

print(x, y)
print(x1, y1)
plt.quiver((0, 0), (0, 0), (x, x1), (y, y1), angles='xy', scale_units='xy', scale=1)
plt.xlabel(comp[1])
plt.ylabel(comp[2])
plt.xlim(0, max(x, y1))
plt.ylim(0, max(y, y1))
plt.show()

from classicAttention import classicAttention
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import torch

device = torch.device("cuda")

glove = WordEmbeddings('Glove')
test_input = Sentence("Apple has AI.")
glove.embed(test_input)


test = classicAttention(5, 100)

base = torch.zeros(100)
base = base.to(device)
# now check out the embedded tokens.
for token in test_input:
    base += token.embedding.to(device)

final_input = base / 3
final_input = final_input.view(1, 100)
print(final_input)
test = test.to(device)
test.forward(final_input)
from flair.embeddings import WordEmbeddings
from flair.data import Sentence
import torch


"""
wordEmbedding

A class used to process our Tweet inputs. Extensible to many kinds of embeddings, a simple
shell built on top of the flair library. 

In the paper, the primary embedding used was Glove.
There are other embeddings that are employed more specifically to Tweet data (WordEmbeddings('twitter')).

In terms of data prep:
    We want to store all of the embeddings in a text file, one on top of the other. 
    So this class should merely exist to execute the embeddings

Args:

    embedding:
        The embedding argument represents the type of embedding that will be used. Varies depending on 
        intended function, and the training set.
        Typically:
            'glove'
            'twitter'

    mode:
        The manner in which varying input lengths will be processed. Will the return vector be a pointwise max, min,
        or an average of all of the input vectors?

    input: 
        The input sentence
"""
class wordEmbedding():
    def __init__(self, embedding:str, mode:str):
        self.embedding = WordEmbeddings(embedding)
        self.mode = mode

    """
    embed
        This function will actually produce the embedding for the input vector.
    """
    def embed(self, input):
        base = torch.zeros(100)
        sentence = Sentence(input)
        self.embedding.embed(sentence)
        # if the mode is maximum, we process a pointwise maximum for the input vectors
        # so we have to return a vector where each value is the max of each column
        if(mode == 'max'):
            # set base to the initial tensor to be processed
            base = sentence[0]
            for token in sentence:
                base = torch.cat((base, token), 0)
            # then we take the "columwise" maximum
            base_t = torch.transpose(base, 0, 1)
            maxed = []
            for tensor in base_t:
                maxed.append(torch.max(tensor))
            return torch.tensor(maxed)
        elif(mode == 'min'):
            for token in sentence:
                base = torch.cat((base, token), 0)
            # then we take the "columwise" maximum
            base_t = torch.transpose(base, 0, 1)
            maxed = []
            for tensor in base_t:
                maxed.append(torch.min(tensor))
            return torch.tensor(maxed)
        elif(mode == 'average'):
            count = 0 
            for token in sentence:
                count += 1
                base += token
            return base / count
        else: 
            raise ValueError("Unsupported argument. Please use an invalid embed mode.")
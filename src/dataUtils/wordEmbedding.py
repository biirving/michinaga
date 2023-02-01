from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings
from flair.data import Sentence
import torch


"""
wordEmbedding (Should it be trained over time)

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

# why don't stacked word embeddings work???
class wordEmbedding():
    def __init__(self, embedding:str, mode:str, stacked:bool):
        self.stacked = stacked
        if(self.stacked):
            # are these embeddings depreciated? what is going on
            self.embeddings = StackedEmbeddings([
                                                    WordEmbeddings(embedding),
                                                    FlairEmbeddings('news-foward'),
                                                    FlairEmbeddings('news-backward'),
                                                    ])
        else:
            self.embedding = WordEmbeddings(embedding)
        self.mode = mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    """
    embed
        This function will actually produce the embedding for the input vector.
    """
    def embed(self, input):
        base = torch.zeros(100)
        base = base.to(self.device)
        sentence = Sentence(input)
        self.embedding.embed(sentence)
        # if the mode is maximum, we process a pointwise maximum for the input vectors
        # so we have to return a vector where each value is the max of each column
        if(self.mode == 'max'):
            base = None
            for token in sentence:
                # set base to the initial tensor to be processed
                if(base is None):
                    base = token.embedding
                else: 
                    base = torch.stack((base, token.embedding))
            # then we take the "columwise" maximum
            base_t = torch.transpose(base, 0, 1)
            maxed = []
            for tensor in base_t:
                maxed.append(torch.max(tensor))
            return torch.tensor(maxed)
        elif(self.mode == 'min'):
            base = None
            for token in sentence:
                # set base to the initial tensor to be processed
                if(base is None):
                    base = token.embedding
                else: 
                    base = torch.stack((base, token.embedding))
            # then we take the "columwise" maximum
            base_t = torch.transpose(base, 0, 1)
            minned = []
            for tensor in base_t:
                minned.append(torch.min(tensor))
            return torch.tensor(minned)
        elif(self.mode == 'average'):
            count = 0 
            for token in sentence:
                count += 1
                toadd = token.embedding
                base += toadd.to(self.device)
            return base / count
        else: 
            raise ValueError("Unsupported argument. Please use an invalid embed mode.")


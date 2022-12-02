"""
Lets train
"""

import torch
from torch import tensor, nn
import matplotlib
from michinaga.src import teanet
from tqdm import tqdm



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def train(model, params):

    """
    A simple training loop. The aim is to train teanet over a specified number of epochs, and to measure accuracy with a basic measure.

    args: 
        - model
            the teanet model to be trained

        - params 
            the parameters to construct the training process
    """

    
    x_train_tweets = params['x_tweet_train']
    x_price_train = params['x_price_train']
    y_train = params['y_train']

    x_test_tweets = params['x_tweet_test']
    x_test_price = params['x_price_test']
    y_test = params['y_test']


    epochs = params['epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    model.to(device)
    adam = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCELoss()
    training_loss_over_epochs = []

    """
    TRAIN
    """
    
    for e in tqdm(range(epochs)):
        train_index = 0
        training_loss = []
        for t in tqdm(range(int(len(x_train_tweets) - 1))):
            model.zero_grad()
            x_input = [x_train_tweets[train_index:train_index+batch_size].view(batch_size, 5, 100).to(device), x_price_train[train_index:train_index+batch_size].view(batch_size, 5, 4).to(device)]
            out = model.forward(x_input)
            loss = loss_fn(out.view(batch_size, 2).float(), y_train[train_index:train_index+batch_size].float().to(device))
            training_loss.append(loss.item())
            adam.zero_grad()
            loss.backward()
            adam.step()
            train_index += batch_size
        
        print('epoch: ', e)
        print('loss total: ', sum(training_loss))
        training_loss_over_epochs.append(training_loss)

    torch.save(model, 'trained_teanet.pt')
    
    """
    EVALUATE

    For the first trial, I just want to see the basic accuracy. The more acute measurements can be implemented later. 
    """
    model = torch.load('trained_teanet.pt')
    model.eval()

    with torch.no_grad():
        num_correct = 0
        tot = 0
        model.setBatchSize(1)
        for y in range(y_test.shape[0]):
            print(y)
            out = model.forward([x_test_tweets[y].view(1, 5, 100).to(device), x_test_price[y].view(1, 5, 4).to(device)])
            actual = torch.max(y_test[y].to(device), dim = 0).indices
            out_index = torch.max(out, dim = 2).indices
            if(actual.item() == out_index.item()):
                num_correct += 1
            tot += 1

    accuracy = num_correct / tot
    print("Basic accuracy on test set: ", accuracy)
    return training_loss_over_epochs, accuracy

if __name__ == "__main__":

    batch_size = 5
    model = teanet(5, 100, 2, batch_size, 5)

    params = {
        'x_tweet_train': torch.load('x_train_tweets.pt'),
        'x_price_train': torch.load('x_train_prices.pt'), 
        'y_train': torch.load('y_train.pt'),
        'x_tweet_test': torch.load('x_test_tweets.pt'),
        'x_price_test': torch.load('x_test_prices.pt'),
        'y_test': torch.load('y_test.pt'),
        'batch_size': batch_size,
        'epochs': 10,
        'learning_rate': 1e-3
    }

    train(model, params)
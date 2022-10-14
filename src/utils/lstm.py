
"""
The LSTM for the TEANET model

The alternative would be to process the price inputs through a transformer like mechanism

"""

# Why an LSTM, versus a decoder? 
# experiment...
class LSTM(nn.Module):
    # do we need the dimensional input
    def __init__(self, dim, batch_size, input):
        super().__init__()

        self.batch_size = batch_size
        self.dim = dim

        # both of these gates recieve a tuple of the x inputs coupled with the hidden states
        # the linear layer automatically initializes the bias
        self.forget_gate = nn.Sequential(nn.Linear(dim), nn.Sigmoid())

        # input gate
        self.input_gate = nn.Sequential(nn.Linear(dim), nn.Sigmoid())

        # initialize this value to 1. Represents the cell state at the previous timestep
        self.c_prior = 1

        # the cell state, the output at each step
        self.cell_state_progress = nn.Sequential(nn.Linear(dim), nn.Tanh())

        # in the lstm, there are the previous values involved in the computation
        self.cell_state_actual = 

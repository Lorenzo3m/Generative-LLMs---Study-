import torch
import torch.nn as nn

# Tensor (datatype) - a matrix or an array, any dimension, any data

# They carry derivatives under the hood, they do the ugly math

a = torch.ones(3, 5) # Creates a 3x5 Tensor (3 rows, 5 collumns)
# print(a)


# SUM AND MEAN

sum_rows = torch.sum(a, axis = 1) # sums the values on the rows [5, 5, 5]
sum_collumns = torch.sum(a, axis = 0) # sums the values on the collumns [3, 3, 3, 3, 3]
#print(sum_collumns)
#print(sum_rows)

# SQUEEZE AND UNSQUEEZE

a = torch.ones(5, 1)
#print(a.shape)   # A 5x1 tensor is basically a 5, the dimension = 1 is not very necessary
squeezed = torch.squeeze(a)
#print(squeezed.shape)  # when we squeeze it the unnecessary dimension gets thrown away

unsqueezed = torch.unsqueeze(squeezed, dim = 1)  # The reverse proccess, this makes it back 5x1.
# if the same was done but dim = 0, it would have made 1x5 tensor



# Neural Network models

class my_model(nn.Module):

    # Constructor
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Linear(4, 6)   # means that, in the first layer we have 4 inputs 
        #  and they each have 6 weights each connected to one of the 6 outputs of the next layer

        self.second_layer = nn.Linear(6, 6) # Second layer receives 6 from the first layer and 
        # gives 6 to the next layer

        self.final_layer = nn.Linear(6, 2) # this model down projects from the second layer to only
        # 2 outputs

    # Forward - get|_model_prediction(example data)
    def forward(self, x): # x are the data points

        #since the forward method is already impplemented in pytorch we need only to call it
        # you can call by doing: first_layer_output = self.first_layer.forward(x) or
        #first_layer_output = self.first_layer(x)

        return self.final_layer(self.second_layer(self.first_layer(x)))

model = my_model()
example_datapoint = torch.randn(1, 4)  # creates a random 1x4 tensor with random values
model(example_datapoint)
#print(model(example_datapoint))

# At this point model returns an uninterpretable results, we have to TRAIN the model
# for some number of iterations

# Then we can actually used it

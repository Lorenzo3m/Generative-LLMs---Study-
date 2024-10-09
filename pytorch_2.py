import torch.nn
from torchtyping import TensorType


# Check documentation for each function used 
#https://pytorch.org/docs/stable/generated/torch.reshape.html
#https://pytorch.org/docs/stable/generated/torch.mean.html
#https://pytorch.org/docs/stable/generated/torch.cat.html
#https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html



class solution:
    def reshape(self, to_reshape):  # function that reshapes a tensor into a new given dimensions
    # torch.reshape() will be very useful

        return torch.reshape(to_reshape, (-1, 2)) # With the -1 pytorch will infer the value to inserted there

    def avg(self, to_avg): # avg in a tensor
    # torch.mean() will be useful

        return torch.mean(to_avg, dim=0) # dim=0 means avg of each collumn

    def concatenate(self, to_concat1, to_concat2): # concatanete 2 tensors
    # torch.cat() will be very useful

        return torch.cat((to_concat1, to_concat2), dim=1) # dim=1 means we are concatenating them left to right 
            # and not stacking them, which would be dim=0

    def get_loss(self, prediction, target): # get the loss in a model. The error function 
    # torch.nn.functional.mse_loss() will be very useful
    # mse = mean squared error

        return torch.nn.functional.mse_loss(prediction, target) # parallel processing, way better than for looping

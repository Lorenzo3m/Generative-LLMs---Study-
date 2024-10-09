import torch
import torch.nn as nn
from torchtyping import TensorType

# Given a squared image with black and white pixel. The white pixel will form a handwritten number.
# The objective of this model is to predict which number has been written in the image and return the 
# correct value.
# Image has a total of 784 pixels, first neural layer will have 784 pixels and the second will have 512
# last layer will have 10 nodes, each node will mean a number.


class num_predict_model(nn.Module):
    def __init__(self):
        super().__init__()

        torch.manual_seed(0) # Sets the seed for generating random numbers on all devices.
            # Returns a torch.Generator object.

        self.first_layer = nn.Linear(784, 512) # Going from one layer to another
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2) # Deactivates nodes with this probability in training
            # This makes the model a little dumber, but that prevents overfitting.
            # Overfitting is when model's training predictions are more accurate than with the data
            # This happens when the model is too complex and considers unuseful parameter in training

        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid() # this ensures outputs from the projection layer are between 0 and 1



    def forward(self, images: TensorType[float] -> TensorType[float]):
        torch.manual_seed(0)
        # Return models prediction with 4 decimal places

        first = self.first_layer(images) # now we want to pass into the ReLU function
        second = self.relu(first) # now we want to pass into the dropout function
        third = self.dropout(second) # now we move to the next layer
        fourth = self.projection(third) # now we pass into the sigmoid function and its done
        out = self.sigmoid(fourth)

        # Note: we could have done all of this in one line of code joining everything, but, 
        # since this is for learning material this way is more understandable, for me

        # If we did all in one line: self.sigmoid(self.projection(self.dropout(self.relu(self.first_layer(images)))))

        return torch.round(out, decimals=4)


model = num_predict_model() # instance of my model

loss_function = nn.CrossEntropyLoss() # we are using this function because our output is in probabiliy
# and not in real values like, 9 or smth
# This does the derivatives for us

optimizer = torch.optim.Adam(model.parameters()) # this does the gradient descent for us
# adam is gradient descent on steroids basically

epochs = 5 # number of times it will run the training, too many mmight not be good
for epoch in range(epochs):
    for images, labels in train_dataloader: # iterator given tuples, one is the image and the labels are the results
        images = images.view(images.shape([0], 784)) # makes the image instead of 28x28 1x784

        # TRAINING BODY

        model_prediction = model(images) # gets the result from our model, calls the forward method
        optimizer.zero_grad() # cancels out the previous derivatives so we can calculate new ones
        loss = loss_function(model_prediction, labels) # gets the error from our current prediction
        loss.backward() # calculates all derivatives necessary to perfom gradient descent, very computational intense
        optimizer.step() # this updates all of the weights: new_w = old_w - derivative*learning_rate
 
### After training you can evaluate the model ####

model.eval()

for images, labels in test_dataloader():
    images = images.view(images.shape[0], 784)

    model_prediction = model(images)
    mas, idx = torch.max(model_prediction, dim = 1) # gets the maximum probability, and its index

    for i in range(len(images)):
        plt.imshow(images[i].view(28, 28)) # printing the images reshaping them back to 28x28
        plt.show()
        print(idx[i].item())
    break


# code not exaclty runnable since the dataloader was not implemented and a file with the images were also no attached.

# But the idea is this

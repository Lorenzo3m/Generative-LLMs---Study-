import torch
import torch.nn as nn
from torchtyping import TensorType

'''
Exercise: 
For the model architecture, first use an embedding layer of size 16. 
Compute the average of the embeddings to remove the time dimension, and end with a single-neuron linear layer followed by a sigmoid. 
The averaging is called the "Bag of Words" model in NLP.

Implement the constructor and forward() pass that outputs the model's prediction as a number between 0 and 1 (completely negative vs. completely positive).
Do not train the model.

Results will naturally be not accurate as we will not train the model
'''


class Sentiment_analysis(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size, 16) # First layer is called the embedding layer, transforming words/tokens into numbers
          # with a lookup table - embedding vector trained with gradient descent
        self.linear_layer = nn.Linear(16, 1)
        self.sigmoid_layer = nn.Sigmoid()  # So our values are squeezed between 0 and 1

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        embeddings = self.embedding_layer(x)
        averaged = torch.mean(embeddings, axis = 1)
        projected = self.linear_layer(averaged)
        return torch.round(self.sigmoid_layer(projected), decimals=4)

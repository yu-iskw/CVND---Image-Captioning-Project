import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, drop_prob=0.5):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.5):
        ''' Initialize the layers of this model.'''
        super(DecoderRNN, self).__init__()

        # the Embedding layer for captions
        self.embed = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs
        # and outputs hidden states of size hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)

        # the Drop out layer between the LSTM layer and the fully connected layer
        self.dropout = nn.Dropout(drop_prob)

        # the linear layer that maps the hidden state output dimension
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # Arrange inputs
        features = features.unsqueeze(1)
        embeddings = self.embed(captions[:, :-1])
        lstm_inputs = torch.cat((features, embeddings), 1)
        lstm_out, _ = self.lstm(lstm_inputs)
        vocab_outputs = self.hidden2vocab(lstm_out)
        return vocab_outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for i in range(max_len):
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_features = self.hidden2vocab(lstm_outputs.squeeze(1))
            predicted = lstm_features.max(1)[1]
            sentence.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
        return sentence

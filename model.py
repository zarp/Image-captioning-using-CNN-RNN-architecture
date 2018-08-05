import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size 
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.word_embeds = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.2) 
        #note: batch_first = True is important
        self.lin1 = nn.Linear(hidden_size, vocab_size) #in:RNN_output->out:Scores
    
    def forward(self, features, captions):
        embeds = self.word_embeds(captions[:, :-1]) #'-1' to get rid of the last <end> which is not used to predict anything
        features = features.unsqueeze(1) #dim of one is inserted at pos = 1
        
        lstm_inputs = torch.cat((features, embeds), 1) #concatenates 'features' and 'embeds' in the given dim (=1)
        lstm_out,  hidden_state = self.lstm(lstm_inputs)
        scores = self.lin1(lstm_out)
        return scores
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # output type: list
        out_tokens = []
        for _ in range(max_len):
            out, states = self.lstm(inputs, states) 
            out = self.lin1(out.squeeze(1))
            maxind = out.max(1)[1]
            out_tokens.append(maxind.item())
            inputs = self.word_embeds(maxind).unsqueeze(1)
            
        return out_tokens
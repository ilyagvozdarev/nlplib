import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
            self
        ):
        super(Attention, self).__init__()


class BahdanauAttention(Attention):

    def __init__(
            self, 
            hidden_size, 
            encoder_hidden_size, 
            alignment_vector_size
        ):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.alignment_vector_size = alignment_vector_size

        self.Wa = nn.Linear(hidden_size, alignment_vector_size)    
        self.Ua = nn.Linear(encoder_hidden_size, alignment_vector_size)
        self.Va = nn.Linear(alignment_vector_size, 1)


    def forward(self, query, keys):
        '''
        query - decoder hidden
        keys - encoder hiddens
        '''
        # print('query.shape = ', query.shape)
        # print('keys.shape = ', keys.shape)
        # query.shape =  torch.Size([B, hidden_size])
        # keys.shape =  torch.Size([B, L_encoder, hidden_size])
        # print('self.Wa(query).shape = ', self.Wa(query).shape)
        # print('self.Ua(keys).shape = ', self.Ua(keys).shape)
        query = query.unsqueeze(1)
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        # print('scores.shape = ', scores.shape)
        scores = scores.squeeze(2)
        # scores.shape =  torch.Size([B, L_encoder])
        weights = F.softmax(scores, dim=-1)
        weights = weights.unsqueeze(1)
        # print('weights.shape = ', weights.shape)
        # print('keys.shape = ', keys.shape)
        context = torch.bmm(weights, keys)
        context = context.squeeze(1)

        #print('Bagdanau')
        # print('context=', context.shape)

        return context, weights
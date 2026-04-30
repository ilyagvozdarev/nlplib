import torch.nn.functional as F
import torch
import torch.nn as nn

from .Attentions import Attention


class EncoderLSTM(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            output_dim,
            bidirectional = True
        ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        #self.fc = nn.Linear(hidden_dim * 2, output_dim)  # * 2 для конкатенации hidden states
        

    def forward(self, seqs_input, lengths):

        # print('self.embedding = ', self.embedding, flush=True)
        # print('seqs_input.shape = ', seqs_input.shape, flush=True)
        # print('seqs_input = ', seqs_input, flush=True)
        # print('self.embedding(seqs_input) = ', self.embedding(seqs_input), flush=True)

        embedded = self.embedding(seqs_input)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_output, (hidden, cell) = self.bilstm(packed_embedded)
        unpacked_output = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)[0]

        # конкатенируем последние hidden и cell state с каждого направления последнего слоя для 
        # входного hidden и cell state для attn_decoder для каждого направления
        output_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output_cell = torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)

        #print('unpacked_output=', unpacked_output.shape)
        return unpacked_output, (output_hidden, output_cell)


class AttnDecoderLSTM(nn.Module):

    def __init__(
            self, 
            vocab_size, 
            embedding_dim,
            hidden_size, 
            output_size,
            attention: Attention,
            bidirectional = True,
            input_hidden_type = 'input+context_hidden'
        ):
        '''
            input_hidden_type:
                тип входа и входного скрытого вектора:
                'input_context':
                    вход = как есть
                    скрытый вектор = вектор контекста для query (query - выходной скрытый вектор с прошлого шага)
                'input+context_hidden':
                    вход = вход конкатенированный с вектором контекста для query (query - скрытый вектор с прошлого шага)
                    скрытый вектор = выходной скрытый вектор с прошлого шага
        '''
        
        super(AttnDecoderLSTM, self).__init__()

        self.input_hidden_type = input_hidden_type

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = attention

        if input_hidden_type == 'input+context_hidden':
            input_size = embedding_dim + attention.encoder_hidden_size
        elif input_hidden_type == 'input_context':
            input_size = embedding_dim
        else:
            raise RuntimeError(f'Неизвестный input_hidden_type = {input_hidden_type}')

        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            batch_first=True,
            bidirectional=bidirectional
        )   

        # вход - конкатенация hidden с каждого направления
        self.out = nn.Linear(hidden_size * 2, output_size)                         


    def forward(
            self, 
            encoder_outputs, 
            init_hidden_forward,
            init_hidden_backward,
            init_cell_forward,
            init_cell_backward,
            input
    ):

        batch_size = len(input)

        def init_hidden_cell():
            return torch.zeros(1, batch_size, self.hidden_size, device = self.device)

        if init_hidden_forward == None or init_hidden_backward == None:
            init_hidden_forward = init_hidden_cell()
            init_hidden_backward = init_hidden_cell()
        if init_cell_forward == None or init_cell_backward == None:
            init_cell_forward = init_hidden_cell()
            init_cell_backward = init_hidden_cell()

        attentions_forward = []
        attentions_backward = []
        forward_hidden = init_hidden_forward
        backward_hidden = init_hidden_backward
        forward_cell = init_cell_forward
        backward_cell = init_cell_backward

        MAX_SEQ_LENGTH = input.shape[1]
        output = None


        def step(
            input,
            hidden,
            cell
        ):
            embedded = self.embedding(input)
            query = hidden
            context, attn_weights = self.attention(query, encoder_outputs)

            # print(f'\nembedded.shape = {embedded.shape}\n')
            # print(f'\ncontext.shape = {context.shape}\n')

            if self.input_hidden_type == 'input_context':
                input_lstm = embedded
                hidden = context
            elif self.input_hidden_type == 'input+context_hidden':
                input_lstm = torch.cat((embedded, context), dim=-1)
            else:
                raise RuntimeError(f'Неизвестный input_hidden_type = {self.input_hidden_type}')

            # print(f'\ninput_lstm.shape = {input_lstm.shape}\n')
            # print(f'\nhidden.shape = {hidden.shape}\n')
            input_lstm = input_lstm.unsqueeze(1)

            _, (hidden, cell) = self.lstm(
                input_lstm, 
                (
                    # конкатенируем так как nn.LSTM в случае bidirectional=True требует 
                    # задавать hidden и cell для обоих направлений, но нам нужно только одно направление
                    # (прямое или обратное) поэтому второй hidden, cell задаем любым (например просто повторяем)
                    torch.cat((hidden.unsqueeze(0), hidden.unsqueeze(0)), dim=0), 
                    torch.cat((cell.unsqueeze(0), cell.unsqueeze(0)), dim=0)
                )
            )

            return hidden, cell, attn_weights


        for i in range(MAX_SEQ_LENGTH):
            # print('input = ', input)
            # print('input[:, i] = ', input[:, i])
            # print()
            # print('forward_hidden = ', forward_hidden)
            # print('forward_hidden.shape = ', forward_hidden.shape)
            # print()

            forward_input = input[:, i]
            backward_input = input[:, -(i + 1)]

            hidden, cell, attn_weights_forward = step(forward_input, forward_hidden, forward_cell)
            # нужен только output hidden прямого прохода, так как вызываем forward 
            # поэлементно с разных концов (вычисления обратного прохода для 
            # i-го элемента не нужны)
            forward_hidden, forward_cell = hidden[-2, :, :], cell[-2, :, :]

            hidden, cell, attn_weights_backward = step(backward_input, backward_hidden, backward_cell)
            # нужен только output hidden прямого прохода, так как вызываем forward 
            # поэлементно с разных концов (вычисления прямого прохода для 
            # -(i + 1)-го элемента не нужны)
            backward_hidden, backward_cell = hidden[-1, :, :], cell[-1, :, :]

            attentions_forward.append(attn_weights_forward)
            attentions_backward.append(attn_weights_backward)
        
        
        # конкатенация последних hidden с каждого направления
        last_hidden = torch.cat((forward_hidden, backward_hidden), dim=-1)  
        output = self.out(last_hidden)

        # print('last_hidden.shape = ', last_hidden.shape)
        # print('output.shape = ', output.shape)

        attentions_forward = torch.cat(attentions_forward, dim=1)
        attentions_backward = torch.cat(attentions_backward, dim=1)

        return (
            output,
            forward_hidden,
            backward_hidden,
            attentions_forward,
            attentions_backward
        )
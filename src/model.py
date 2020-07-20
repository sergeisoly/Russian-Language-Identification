import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math

from src.data import PAD_TOKEN


class GRUIdentifier(nn.Model):
    def __init__(self, vocab_size : int, n_classes : int, embedding_dim : int,
                 hidden_dim : int, bidirectional : bool, dropout_p : float,
                 **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim,
                          bidirectional=bidirectional, batch_first=True)
        h0_tensor = torch.Tensor(1, hidden_dim)
        nn.init.xavier_normal_(h0_tensor, gain=1.)
        self.h_0_init = nn.Parameter(h0_tensor)

        for layer_p in self.gru._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.orthogonal_(self.gru.__getattr__(p))

        if bidirectional:
            self.hidden2label = nn.Linear(2*hidden_dim, n_classes)
        else:
            self.hidden2label = nn.Linear(hidden_dim, n_classes)            

    def init_hidden(self, batch_size : int) -> torch.Tensor:
        # Initialise hidden state with learned hidden state
        h_0 = self.h_0_init.repeat(2 if self.bidirectional else 1, batch_size, 1)

        if torch.cuda.is_available():
            return h_0.cuda()
        else:
            return h_0

    def forward(self, sentence : Variable, lengths : torch.Tensor) -> torch.Tensor:

        batch_size = sentence.shape[0]
        x = self.embeddings(sentence)  # batch, time, dim
        x = self.dropout(x)
        packed_x = pack_padded_sequence(x, lengths, batch_first=True)

        # Recurrent part
        hidden_in = self.init_hidden(batch_size)
        recurrent_out, hidden_out = self.gru(packed_x, hidden_in)
        recurrent_out, _ = pad_packed_sequence(recurrent_out, batch_first=True)

        # Unpack packed sequences
        dim = recurrent_out.size(2)
        indices = lengths.view(-1, 1).unsqueeze(2).repeat(1, 1, dim) - 1
        indices = indices.cuda() if torch.cuda.is_available() else indices
        final_states = torch.squeeze(torch.gather(recurrent_out, 1, indices), dim=1)
        # Classification
        y = self.hidden2label(final_states)

        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        # log_probs = F.log_softmax(y, 1)
        log_probs = torch.sigmoid(y)
        return log_probs

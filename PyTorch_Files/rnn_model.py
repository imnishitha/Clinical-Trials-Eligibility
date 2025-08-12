
import torch.nn as nn


class RNNClassifierFromScratch(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, num_layers: int, num_labels: int, dropout_rate: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout_classifier = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids) 

        lengths = attention_mask.sum(dim=1)
        
        lengths = lengths.cpu().clamp(min=1) 

        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False 
        )

        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        final_hidden_state = hidden[-1, :, :]
        
        pooled_output = self.dropout_classifier(final_hidden_state)
        
        logits = self.classifier(pooled_output)
        
        return logits
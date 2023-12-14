import torch
import torch.nn as nn
import torch.optim as optim

class SimpleChatbotModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SimpleChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Tomar la última salida de la secuencia
        return output

# Guardar el modelo
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Cargar el modelo
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

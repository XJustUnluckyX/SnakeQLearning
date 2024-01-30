import torch
import torch.nn as Nn
import torch.optim as opt
import torch.nn.functional as Fn
import os


class Linear_QNet(Nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # Inizializzazione dei layer
        super().__init__()
        self.linear1 = Nn.Linear(input_size, hidden_size)
        self.linear2 = Nn.Linear(hidden_size, output_size)

    def forward(self, x):  # Rete neurale feed-forward, nomenclatura necessaria
        x = Fn.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save_model(self, file_name='mdl.pth'):  # Salvataggio del modello
        model_path = './mdl'
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        file_name = os.path.join(model_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.loss = Nn.MSELoss()  # Funzione di perdita MSE
        self.optimizer = opt.Adam(model.parameters(), lr=self.lr)  # Ottimizzatore Adam

    def train(self, state, action, reward, next_state, done):  # Model Trainer
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)  # Usiamo la formula predittiva Q=model

        target = pred.clone()  # Nuovo Q = r+y(prossimo Q previsto)
        for i in range(len(done)):
            Q_next = reward[i]
            if not done[i]:
                Q_next = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_next

        self.optimizer.zero_grad()  # Calcolo della loss function
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()

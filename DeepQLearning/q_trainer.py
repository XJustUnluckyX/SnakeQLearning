import torch.nn as nn
import torch.optim as optim
import torch

# E' stato utilizzato come riferimento il seguente progetto
# https://github.com/patrickloeber/snake-ai-pytorch/tree/main , poi rifattorizzato

class QNetTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

    def train_net(self, state, action, reward, following_state, done):  # Funzione di addestramento della rete
        state = torch.tensor(state, dtype=torch.float)
        following_state = torch.tensor(following_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:  # Calcolo dello stato a partire dal tensore
            state = torch.unsqueeze(state, 0)
            following_state = torch.unsqueeze(following_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        prediction = self.model(state)

        target = prediction.clone()
        for i in range(len(done)):
            Q_nxt = reward[i]
            if not done[i]:
                Q_nxt = reward[i] + self.gamma * torch.max(self.model(following_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_nxt

        self.optimizer.zero_grad()
        loss = self.loss(target, prediction)
        loss.backward()
        self.optimizer.step()

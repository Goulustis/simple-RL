import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

# original code from https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/simple_dqn_T_2020.py

# target network modification:
# https://pythonprogramming.net/training-deep-q-learning-dqn-reinforcement-learning-python-tutorial/
class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

# class PositionalEmbedding(nn.Module):
#     # for intuition https://medium.com/swlh/elegant-intuitions-behind-positional-encodings-dc48b4a4a5d1

#     def __init__(self, d_model, max_len=32):
#         super().__init__()

#         # Compute the positional encodings once in log space.
#         pe = T.zeros(max_len, d_model).float()
#         pe.require_grad = False

#         position = T.arange(0, max_len).float().unsqueeze(1)
#         div_term = (T.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

#         pe[:, 0::2] = T.sin(position * div_term)
#         pe[:, 1::2] = T.cos(position * div_term)

#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         self.to(self.device)

#     def forward(self, x):
#         return self.pe[:, :x.size(1)]

class PositionalEncoding(nn.Module):    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        d_model_use = d_model if d_model%2 == 0 else d_model + 1
        pe = T.zeros(max_len, d_model_use)
        position = T.arange(0, max_len,dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, d_model_use, 2).float()*(-math.log(10000.0) / d_model))        
        pe[:, 0::2] = T.sin(position * div_term)        
        pe[:, 1::2] = T.cos(position * div_term)     

        pe = (pe[:, :d_model]*0.05).float()

        pe = pe.unsqueeze(0)#.transpose(0, 1)        
        self.register_buffer('pe', pe)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        seq_len = x.shape[1]
        # x = x + self.pe[:x.size(0), :]
        x = x + self.pe[:, :seq_len, :]
        return x #self.dropout(x)


class Encoder(nn.Module):
    def __init__(self, d_model=3, hidden_dim=32, max_len=32):
        super().__init__()
        self.no_act = T.zeros(d_model)
        self.no_act[0] = 1.
        self.no_act = self.no_act.reshape(1, d_model)
        self.pos_encod = PositionalEncoding(d_model=d_model)#PositionalEmbedding(d_model=d_model, max_len=32)
        self.max_len = max_len

        self.fc1 = nn.Linear(d_model*max_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def pad(self, x):
        n, f_dim = x.shape

        if n >= self.max_len:
            return x[-self.max_len:, :]

        tmp_x = T.zeros(self.max_len, f_dim)
        tmp_x[:n] = x
        return tmp_x

    def trans_input(self, x):
        if len(x) == 0:
            x_tmp = T.zeros(1,3)
            x_tmp[0,0] = 1
            return x_tmp
        x = T.tensor(x)#.unsqueeze(0)
        n = len(x)
        x = x + 1
        idxs = list(range(n))
        x_tmp = T.zeros(n, 3)
        x_tmp[idxs, x] = 1
        return x_tmp 
    
    def process_input(self, x):
        trans_inp = self.trans_input(x)
        pad_inp = self.pad(trans_inp)
        return pad_inp

    def forward(self, x, processed = True):
        # processed --- is the input processed already?
        if not processed:
            x = self.trans_input(x)
            x = self.pad(x)
            if len(x) != 3:
                x = x.unsqueeze(0).to(self.device)
        x = self.pos_encod(x) #x + self.pos_encod(x)
        x = x.reshape(len(x), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x


class Agent:
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, enc_h_dim=128,
                 player_code = "1", state_len = 32):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 100
        self.player_code = str(player_code)

        self.state_len = state_len  # length of other player to consider
        self.enc_h_dim = enc_h_dim  # hidden dim for encoder
        input_dims = (enc_h_dim,)  

        self.encoder = Encoder(n_actions+1, 
                                hidden_dim=self.enc_h_dim, 
                                max_len=state_len)

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        
        # target network
        self.Q_target = DeepQNetwork(lr, n_actions=n_actions,
                                   input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)
        
        self.Q_target.load_state_dict(self.Q_eval.state_dict())


        # self.state_memory = np.zeros((self.mem_size, *input_dims),
        #                              dtype=np.float32)
        # self.new_state_memory = np.zeros((self.mem_size, *input_dims),
        #                                  dtype=np.float32)
        self.state_memory = np.zeros((self.mem_size, *(state_len, n_actions+1)),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *(state_len, n_actions+1)),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = self.encoder.process_input(state).numpy()
        self.new_state_memory[index] = self.encoder.process_input(state_).numpy()
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1



    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # state = T.tensor([observation]).to(self.Q_eval.device)
            state = self.encoder(observation, processed=False)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action



    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        state_batch = self.encoder(state_batch)
        new_state_batch = self.encoder(new_state_batch)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
        

    def update_targ_model(self):
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
    

    def save_models(self, game_iter):
        T.save("checkpoint/player_%s/q_fnc_%s.pth"%(self.player_code, game_iter), self.Q_eval)
        T.save("checkpoint/player_%s/encoder_%s.pth"%(self.player_code, game_iter), self.encoder)


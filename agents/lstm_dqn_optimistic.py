import torch
import numpy as np
import nltk
import random
nltk.download('punkt')
from nltk import word_tokenize

class LSTM_DQN(torch.nn.Module):
  def __init__(self, emb_size, action_space, hidden_dim):
    super(LSTM_DQN, self).__init__()

    self.emb = torch.nn.Embedding(emb_size, hidden_dim)
    self.lstm = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers=1)

    self.dqn = torch.nn.Sequential(
        torch.nn.Linear(emb_size, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, action_space)
    )

    for layer in self.dqn:
      if isinstance(layer, torch.nn.Linear):
        torch.nn.init.constant_(layer.weight, 10000.0)

    self.hidden = None

  def forward(self, input_tensor):
    embeddings = self.emb(input_tensor)
    if not self.hidden:
      output, hidden = self.lstm(embeddings)
    else: 
      output, hidden = self.lstm(embeddings, self.hidden)
    self.hidden = hidden
    state_rep = torch.mean(output, dim=1)
    q_vals = self.dqn(state_rep)
    return q_vals

class LSTM_DQN_Agent():
  def __init__(self, vocab_size, max_num_actions):
    self.word_indices = {'UNK': 0}
    self.last_word_idx = 1
    self.action_indices = {'UNK': 0}
    self.idx_to_action = {0: 'UNK'}
    self.last_action_idx = 1

    self.model = LSTM_DQN(emb_size=vocab_size+max_num_actions, action_space=max_num_actions, hidden_dim=128)
    self.optimizer = torch.optim.Adam(self.model.parameters(), 0.00003)
    self.vocab_size = vocab_size
    self.num_actions = max_num_actions
    self.loss_fcn = torch.nn.MSELoss()

  def vectorize_obs(self, tokens, eval_mode):
    for token in tokens:
      if not eval_mode and token not in self.word_indices.keys():
        self.word_indices[token] = self.last_word_idx
        self.last_word_idx+=1

    if eval_mode:
      word_vec = [self.word_indices[token] if token in self.word_indices else 0 for token in tokens]
    else:
      word_vec = [self.word_indices[token] for token in tokens if token in self.word_indices]

    torch_indices = torch.tensor(word_vec)

    one_hot = torch.nn.functional.one_hot(torch_indices, num_classes=self.vocab_size)
    token_vector = torch.sum(one_hot, dim=0)

    return token_vector

  def tokenize_obs(self, sentence):
    tokens = word_tokenize(sentence)
    return tokens

  def process_obs(self, sentence, eval_mode=False):
    tokens = self.tokenize_obs(sentence)
    vectors = self.vectorize_obs(tokens, eval_mode)
    return vectors

  def process_actions(self, actions, eval_mode=False):
    for action in actions:
      if not eval_mode and action not in self.action_indices.keys():
        self.action_indices[action] = self.last_action_idx
        self.idx_to_action[self.last_action_idx] = action
        self.last_action_idx+=1

    if eval_mode:
      action_vec = [self.action_indices[action] if action in self.action_indices else 0 for action in actions]
    else:
      action_vec = [self.action_indices[action] for action in actions if action in self.action_indices]

    torch_indices = torch.tensor(action_vec)

    one_hot = torch.full((self.num_actions, ), 0)
    one_hot[torch_indices] = 100

    return one_hot

  def get_qvals(self, obs, infos, reward, eval_mode=False):
    obs_tensor = self.process_obs(obs, eval_mode)
    actions_tensor = self.process_actions(infos['admissible_commands'], eval_mode)

    input_tensor = torch.cat((obs_tensor,actions_tensor), 0)

    if eval_mode:
      self.model.eval()
      with torch.no_grad():
        q_vals = self.model(input_tensor)
      self.model.train()
    else:
      q_vals = self.model(input_tensor)

    # valid_cmds = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
    # valid_indices = [self.action_indices.get(cmd, 0) for cmd in valid_cmds]
    # valid_q_vals = q_vals[valid_indices]
    # print('OPTIONS', valid_cmds, valid_q_vals)

    return q_vals

  def get_max_qval(self, q_vals, infos):
    valid_cmds = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
    valid_indices = [self.action_indices.get(cmd, 0) for cmd in valid_cmds]

    valid_q_vals = q_vals[valid_indices]
    max_value = torch.max(valid_q_vals)

    max_cmds = [cmd for cmd, q_val in zip(valid_cmds, valid_q_vals) if q_val == max_value]
    best_action = max_cmds[0]

    return (best_action, max_value)

  def act(self, obs, infos, reward, q_vals, eps_greedy=True, epsilon=0.15):
    action, max_val = self.get_max_qval(q_vals, infos)

    if eps_greedy:
      prob = np.random.random()
      if prob < epsilon:
        options = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
        random_action = random.choice(options)
        q_idx = self.action_indices.get(random_action, -1)
        if q_idx == -1: return (random_action, 0)
        return random_action, q_vals[q_idx]

    return action, max_val
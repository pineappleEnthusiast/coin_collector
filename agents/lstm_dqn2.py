import torch
import numpy as np
import nltk
import random
nltk.download('punkt')
from nltk import word_tokenize

class LSTM_DQN(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super(LSTM_DQN, self).__init__()

    self.hidden_dim = hidden_size
    self.emb = torch.nn.Embedding(input_size, hidden_size)
    self.encoder = torch.nn.LSTM(hidden_size, hidden_size, num_layers=1)
    self.cmd_encoder = torch.nn.LSTM(hidden_size, hidden_size, num_layers=1)
    self.state_encoder = torch.nn.LSTM(hidden_size, hidden_size, num_layers=1)

    self.critic = torch.nn.Linear(hidden_size, 1)
    self.att_cmd = torch.nn.Linear(hidden_size * 2, 1)

    self.hidden = (torch.zeros(1, self.hidden_dim), torch.zeros(1, self.hidden_dim))

  def forward(self, obs, commands):
    input_length = obs.size(0)
    num_commands = commands.size(1)

    embedded = self.emb(obs)
    encoder_output, encoder_hidden = self.encoder(embedded)
    state_output, state_hidden = self.state_encoder(encoder_hidden[0], self.hidden)
    self.hidden = state_hidden
    value = self.critic(state_output)

    cmd_embedding = self.emb.forward(commands)
    _, cmds_encoding_last_states = self.cmd_encoder.forward(cmd_embedding)
    cmd_selector_input = torch.stack([state_hidden[0]] * num_commands, 1)
    cmds_encoding_last_states = torch.stack([cmds_encoding_last_states[0]], 1).squeeze(0)
    cmd_selector_input = torch.cat([cmd_selector_input, cmds_encoding_last_states], dim=-1)

    scores = torch.nn.functional.relu(self.att_cmd(cmd_selector_input)).squeeze(-1)
    return scores.squeeze(0)

class LSTM_DQN_Agent():
  def __init__(self, vocab_size, max_num_actions):
    self.word_indices = {'UNK': 0}
    self.last_word_idx = 1

    self.model = LSTM_DQN(input_size=vocab_size, hidden_size=128)
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
    action_vectors = []
    for action in actions:
      action_vectors.append(self.process_obs(action))

    return torch.stack(action_vectors).T

  def get_qvals(self, obs, infos, reward, eval_mode=False):
    obs_tensor = self.process_obs(obs, eval_mode)
    actions_tensor = self.process_actions(infos['admissible_commands'], eval_mode)

    # input_tensor = torch.cat((obs_tensor,actions_tensor), 0)

    if eval_mode:
      self.model.eval()
      with torch.no_grad():
        q_vals = self.model(obs_tensor, actions_tensor)
      self.model.train()
    else:
      q_vals = self.model(obs_tensor, actions_tensor)

    # valid_cmds = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
    # valid_indices = [self.action_indices.get(cmd, 0) for cmd in valid_cmds]
    # valid_q_vals = q_vals[valid_indices]
    # print('OPTIONS', valid_cmds, valid_q_vals)

    return q_vals

  def get_max_qval(self, q_vals, infos):
    valid_cmds = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
    valid_q_vals = [val for cmd, val in zip(infos['admissible_commands'], q_vals) if cmd in valid_cmds]
    max_value = max(valid_q_vals)

    max_cmds = [cmd for cmd, q_val in zip(valid_cmds, valid_q_vals) if q_val == max_value]
    best_action = max_cmds[0]

    return (best_action, max_value)

  def act(self, obs, infos, reward, q_vals, eps_greedy=True, epsilon=0.15):
    action, max_val = self.get_max_qval(q_vals, infos)

    if eps_greedy:
      prob = np.random.random()
      if prob < epsilon:
        options = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
        valid_qvals = [val for cmd, val in zip(infos['admissible_commands'], q_vals) if cmd in options]
        random_idx = random.randint(0, len(options) - 1)
        return options[random_idx], valid_qvals[random_idx]

    return action, max_val
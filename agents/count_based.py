import torch
import numpy as np
import nltk
nltk.download('punkt')
from nltk import word_tokenize

from .lstm_dqn2 import LSTM_DQN_Agent

class LSTM_DQN_Episodic_Count_Agent(LSTM_DQN_Agent):
  def __init__(self, vocab_size, max_num_actions):
    super().__init__(vocab_size, max_num_actions)
    self.state_counts = {}

  def get_qvals(self, obs, infos, reward, eval_mode=False):
  #   obs_tensor = self.process_obs(obs, eval_mode)
  #   actions_tensor = self.process_actions(infos['admissible_commands'], eval_mode)

  #   input_tensor = torch.cat((obs_tensor,actions_tensor), 0)

  #   if eval_mode:
  #     self.model.eval()
  #     with torch.no_grad():
  #       q_vals = self.model(input_tensor)
  #     self.model.train()
  #   else:
  #     q_vals = self.model(input_tensor)

  #   # if not eval_mode:
  #   #   valid_cmds = [cmd for cmd in infos['admissible_commands'] if cmd not in ['look', 'inventory', 'examine coin']]
  #   #   valid_indices = [self.action_indices.get(cmd, 0) for cmd in valid_cmds]
  #   #   valid_q_vals = q_vals[valid_indices]
  #   #   print('OPTIONS', valid_cmds, valid_q_vals)

  #   count_key = tuple(input_tensor.detach().numpy())

  #   #INTRINSIC REWARD:
  #   if not eval_mode:
  #     if count_key not in self.state_counts: self.state_counts[count_key] = 0
  #     else: self.state_counts[count_key] = -1

  #     return (q_vals, self.state_counts[count_key])

  #   return (q_vals, 0)


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

    return q_vals, 0
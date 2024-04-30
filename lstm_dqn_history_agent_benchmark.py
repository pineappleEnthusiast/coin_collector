import matplotlib as mpl
mpl.use('TkAgg')

import torch
import random
import textworld.gym
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pickle

from agents.lstm_dqn import LSTM_DQN_Agent

random.seed(42)

TRAIN_SEED = 0
TEST_SEED = 42
TRAIN_LEVEL = 10
REWARD_SCALE = 1000
NUM_EPOCHS = 10

print('TRAIN_SEED', TRAIN_SEED)

def make_game(seed, level):
  output = f'tw_games/coin_collector_lev{level}_seed{seed}_game.ulx'
  if not os.path.exists(output):
    command = ['tw-make', 'tw-coin_collector', '--level', str(level), '--seed', str(seed), '--output', output]
    subprocess.run(command)

  request_infos = textworld.EnvInfos(
    admissible_commands=True,  # All commands relevant to the current state.
    entities=True,              # List of all interactable entities found in the game.
    facts=True
  )

  env_id = textworld.gym.register_game(output, request_infos)
  env = textworld.gym.make(env_id)
  return env

TRAIN_ENV = make_game(TRAIN_SEED, TRAIN_LEVEL)
TEST_ENV = make_game(TEST_SEED, TRAIN_LEVEL)

def train(agent, env, max_num_moves, gamma=0.90):
  num_episodes = NUM_EPOCHS
  tot_moves = 0

  goal_value_per_epoch = []
  test_score_per_epoch = [[], []]
  num_moves_per_epoch = [[], []]

  for _ in range(num_episodes):
    obs, infos = env.reset()

    done = False
    obs, score, done, infos = env.step('look')
    obs = ' '.join(obs.split())
    num_moves = 0

    history = ['', '', '', '', obs]

    while num_moves < max_num_moves and not done:
      obs = ''.join(history)
      q_vals = agent.get_qvals(obs, infos, score)
      action, action_qval = agent.act(obs, infos, score, q_vals, eps_greedy=True)

      obs, score, done, infos = env.step(action)
      obs = ' '.join(obs.split())
      obs = action + ' ' + obs

      num_moves += 1
      tot_moves += 1

      score *= REWARD_SCALE
            
      next_qvals = agent.get_qvals(obs, infos, score, eval_mode=True)

      if done: next_action_qval = torch.tensor(0)
      else: next_action, next_action_qval = agent.get_max_qval(next_qvals, infos)

      loss = agent.loss_fcn(action_qval, score + gamma * next_action_qval)
      loss.backward()
      agent.optimizer.step()
      agent.optimizer.zero_grad()

      history.pop(0)
      history.append(obs)

    test_scores, test_num_moves = test_on_train_level(agent, max_num_moves)
    goal_idx = agent.action_indices.get('take coin', 0)
    goal_qval = next_qvals[goal_idx]

    goal_value_per_epoch.append(goal_qval.item())
    test_score_per_epoch[0].append(test_scores[0])
    num_moves_per_epoch[0].append(test_num_moves[0])
    test_score_per_epoch[1].append(test_scores[1])
    num_moves_per_epoch[1].append(test_num_moves[1])

    agent.model.hidden = (torch.zeros(1, agent.model.hidden_dim), torch.zeros(1, agent.model.hidden_dim))

  return agent, goal_value_per_epoch, test_score_per_epoch, num_moves_per_epoch


def test_on_train_level(agent, max_num_moves):
  test_scores = []
  test_moves = []

  for env in [TRAIN_ENV, TEST_ENV]:
    obs, infos = env.reset()

    num_moves = 0
    done = False
    obs, score, done, infos = env.step('look')
    
    while num_moves < max_num_moves and not done:
      q_vals = agent.get_qvals(obs, infos, score, eval_mode=True)
      action, _ = agent.act(obs, infos, score, q_vals, False)
      obs, score, done, infos = env.step(action)
      score *= REWARD_SCALE

      num_moves += 1

    test_scores.append(score)
    test_moves.append(num_moves)

  agent.model.hidden = (torch.zeros(1, agent.model.hidden_dim), torch.zeros(1, agent.model.hidden_dim))
  
  return test_scores, test_moves


def plot_test_on_train(goal_values, test_scores, num_moves, max_num_moves):
  avg_goal_values = np.mean(goal_values, axis=0)
  avg_test_scores = np.mean(test_scores, axis=0)
  avg_num_moves = np.mean(num_moves, axis=0)

  std_goal_values = np.std(goal_values, axis=0)
  std_test_scores = np.std(test_scores, axis=0)
  std_num_moves = np.std(num_moves, axis=0)

  print(std_goal_values, std_test_scores, std_num_moves)

  fig, axs = plt.subplots(1, 3, figsize=(18,5))

  epochs = np.arange(NUM_EPOCHS)

  axs[0].plot(epochs, avg_goal_values)
  axs[0].fill_between(epochs, avg_goal_values - std_goal_values, avg_goal_values + std_goal_values, color='gray', alpha=0.2, label='Standard Deviation')
  axs[0].set_title('Magnitude of Value for TAKE COIN cmd [Level 10]')
  axs[0].set_xticks(epochs)
  axs[0].set_xlabel('Epochs')
  axs[0].set_ylabel('Value of TAKE COIN cmd')

  axs[1].plot(epochs, avg_test_scores[0], label='train game')
  axs[1].plot(epochs, avg_test_scores[1], label='test game')
  axs[1].fill_between(epochs, avg_test_scores[0] - std_test_scores[0], avg_test_scores[0] + std_test_scores[0], color='gray', alpha=0.2, label='Train Standard Deviation')
  axs[1].fill_between(epochs, avg_test_scores[1] - std_test_scores[1], avg_test_scores[1] + std_test_scores[1], color='blue', alpha=0.2, label='Test Standard Deviation')
  axs[1].set_title('# Score for 1 Play of Test Games [Level 10]')
  axs[1].set_xticks(range(0, NUM_EPOCHS + 1, 10))
  axs[1].set_xlabel('Epochs')
  axs[1].set_ylabel(f'Game Score')
  axs[1].legend()

  axs[2].plot(epochs, avg_num_moves[0], label='train game')
  axs[2].plot(epochs, avg_num_moves[1], label='test game')
  axs[2].fill_between(epochs, avg_num_moves[0] - std_num_moves[0], avg_num_moves[0] + std_num_moves[0], color='gray', alpha=0.2, label='Train Standard Deviation')
  axs[2].fill_between(epochs, avg_num_moves[1] - std_num_moves[1], avg_num_moves[1] + std_num_moves[1], color='blue', alpha=0.2, label='Test Standard Deviation')
  axs[2].set_title('# Moves for 1 Play of Test Games [Level 10]')
  axs[2].set_xticks(range(0, NUM_EPOCHS + 1, 10))
  axs[2].set_xlabel('Epochs')
  axs[2].set_ylabel(f'# Moves out of {max_num_moves}')
  axs[2].legend()

  plt.tight_layout()
  plt.savefig('LSTM_DQN_HISTORY.png')
  plt.show()


# Let's train the agent now
quest_length = ((TRAIN_LEVEL - 1) % 100 + 1)
max_num_moves = 50 # quest_length * 2

all_goal_values = []
all_test_scores = []
all_num_moves = []

NUM_ABLATIONS = 10
for i in range(NUM_ABLATIONS):
  print('ablation', i)
  untrained_baseline = LSTM_DQN_Agent(vocab_size=10000, max_num_actions=1000)
  trained_baseline = train(untrained_baseline, TRAIN_ENV, max_num_moves)
  trained_agent, goal_values, test_scores, test_num_moves = trained_baseline
  all_goal_values.append(np.array(goal_values))
  all_test_scores.append(np.array(test_scores))
  all_num_moves.append(np.array(test_num_moves))

save_data = {'all_goal_values': all_goal_values, 'all_test_scores': all_test_scores, 'all_num_moves': all_num_moves}
with open('lstm_dqn_history_data.pkl', 'wb') as f:
  pickle.dump(save_data, f)

plot_test_on_train(all_goal_values, all_test_scores, all_num_moves, max_num_moves)
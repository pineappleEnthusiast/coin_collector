import matplotlib as mpl
mpl.use('TkAgg')

import textworld.gym
import os
import subprocess
import matplotlib.pyplot as plt

from agents.count_based import LSTM_DRQN_Agent

TRAIN_SEED = 0
TEST_SEEDS = [74, 10, 91, 77]
TRAIN_LEVEL = 10
REWARD_SCALE = 1000
NUM_EPOCHS = 10

print('TRAIN_SEED', TRAIN_SEED)
print('TEST_SEEDS', TEST_SEEDS)

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
  return env_id

TRAIN_ENV = make_game(TRAIN_SEED, TRAIN_LEVEL)
TEST_ENVS = [make_game(seed, TRAIN_LEVEL) for seed in TEST_SEEDS]

print('TRAIN_ENV ID', TRAIN_ENV)
print('TEST_ENVS IDS', TEST_ENVS)

def train(agent, env_id, max_num_moves, gamma=0.9):
  num_episodes = NUM_EPOCHS
  losses = []

  for _ in range(num_episodes):
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()

    done = False
    obs, score, done, infos = env.step('look')
    num_moves = 0
    running_loss = 0.0

    # selected_qvals = []
    # target_qvals = []

    while num_moves < max_num_moves and not done:

      q_vals = agent.get_qvals(obs, infos, score)
      action, action_qval = agent.act(obs, infos, score, q_vals, True)

      obs, score, done, infos = env.step(action)

      score *= REWARD_SCALE

      next_qvals = agent.get_qvals(obs, infos, score)
      _, next_action_qval = agent.get_max_qval(next_qvals, infos)

      loss = agent.loss_fcn(action_qval, score + gamma * next_action_qval)
      loss.backward()
      agent.optimizer.step()
      agent.optimizer.zero_grad()

      num_moves += 1
      running_loss += loss.detach().item()

    #   selected_qvals.append(action_qval)
    #   target_qvals.append(score + gamma * next_action_qval)

    # loss = agent.loss_fcn(torch.stack(selected_qvals), torch.stack(target_qvals))
    # loss.backward()

    # agent.optimizer.step()
    # agent.optimizer.zero_grad()

    avg_loss = running_loss / num_moves

    losses.append(avg_loss)
    print('AVG LOSS:', losses[-1])

    test_on_train_level(agent, max_num_moves)

  return agent, losses

# Each game was generated with a fixed seed, so the game dynamics do not change
# At test time, we make the agent greedy w.r.t. to the learned value function, so the policy used is also deterministic
# Accordingly, it's sufficient to only collect stats for one run of the agent on each game because the results can't change across runs

def test_on_train_level(agent, max_num_moves):
  all_test_envs = [TRAIN_ENV] + TEST_ENVS

  all_test_scores = [0] * len(all_test_envs)
  all_num_moves = [0] * len(all_test_envs)

  for i, env_id in enumerate(all_test_envs):
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()

    done = False
    obs, score, done, infos = env.step('look')
    num_moves = 0

    while num_moves < max_num_moves and not done:
      q_vals = agent.get_qvals(obs, infos, score, eval_mode=True)
      action, _ = agent.act(obs, infos, score, q_vals, False)

      obs, score, done, infos = env.step(action)
      score *= REWARD_SCALE

      num_moves += 1

    all_test_scores[i] = score
    all_num_moves[i] = num_moves

  print('TEST SCORES:', all_test_scores)
  print('MOVES MADE BEFORE END OF EPISODE', all_num_moves)

  return all_test_scores, all_num_moves

def plot_test_on_train(all_test_scores, all_num_moves, max_num_moves):
  all_test_seeds = [TRAIN_SEED] + TEST_SEEDS
  games = [str(seed) for seed in all_test_seeds]

  fig, axs = plt.subplots(1, 2, figsize=(12,5))
  axs[0].bar(games, all_test_scores)
  axs[0].set_title('Scores for 1 Play of Each Test Game [Level 10]')
  axs[0].set_xlabel('Game Seed')
  axs[0].set_ylabel('Game Score')

  axs[1].bar(games, all_num_moves)
  axs[1].set_title('# Moves for 1 Play of Each Test Game [Level 10]')
  axs[1].set_xlabel('Game Seed')
  axs[1].set_ylabel(f'# Moves out of {max_num_moves}')

  plt.tight_layout()
  plt.savefig('LSTM_DQN.png')
  plt.show()

def plot_loss(losses, num_epochs):
  epochs = list(range(0, num_epochs))
  plt.plot(epochs, losses)
  plt.title('Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Batched Loss')

  plt.grid(True)
  plt.savefig('LSTM_DQN_LOSS.png')
  plt.show()

# Let's train the agent now
untrained_baseline = LSTM_DRQN_Agent(vocab_size=10000, max_num_actions=1000)
quest_length = ((TRAIN_LEVEL - 1) % 100 + 1)
max_num_moves = quest_length * 2

trained_baseline = train(untrained_baseline, TRAIN_ENV, max_num_moves)
trained_agent, losses = trained_baseline

plot_loss(losses, NUM_EPOCHS)

all_test_scores, all_num_moves = test_on_train_level(trained_agent, max_num_moves)
plot_test_on_train(all_test_scores, all_num_moves, max_num_moves)
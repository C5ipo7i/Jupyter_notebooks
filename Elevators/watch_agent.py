import time 
import numpy as np
from collections import deque

from hotel import Hotel
from config import Config
from ddpg import DDPG
from train import seed_replay_buffer,train_ddpg,into_action,deterministic_actions
from networks import Actor,Critic

def watch(algo):
    N_floors = 15
    N_elevators = 2
    N_people = 2
    # Load the ENV
    env = Hotel(N_floors,N_elevators,N_people)

    # size of each action
    action_space = env.action_space

    # examine the state space 
    state_space = env.state_space
    print('Size of each action: {}, Size of the state space {}'.format(action_space,state_space))
    
    config = Config(algo)

    agent = DDPG(state_space, action_space,Actor,Critic,config)
    agent.load_weights(config.checkpoint_path)
    # Fill buffer with random actions up to min buffer size
    printing = True
    episodes,tmax = config.episodes,config.tmax
    tic = time.time()
    means = []
    mins = []
    maxes = []
    stds = []
    mean_steps = []
    steps = []
    scores_window = deque(maxlen=100)
    for e in range(1,2):
        agent.reset_episode()
        episode_scores = []
        obs = env.reset()
        for t in range(tmax):
            probs = agent.act(obs)
            print('agent probas',probs)
            # actions = into_action(probs)
            actions = deterministic_actions(probs)
            next_obs,rewards,dones = env.step(actions,printing)
            # Score tracking
            episode_scores.append(rewards)
            obs = next_obs
            if dones:
#                 print('done',t)
                steps.append(int(t))
                break
            
        scores_window.append(np.sum(episode_scores))
        means.append(np.mean(scores_window))
        mins.append(min(scores_window))
        maxes.append(max(scores_window))
        mean_steps.append(np.mean(steps))
        stds.append(np.std(scores_window))
        toc = time.time()
        r_mean = np.mean(scores_window)
        r_max = max(scores_window)
        r_min = min(scores_window)
        r_std = np.std(scores_window)
        print("\rEpisode: {} out of {}, Steps {}, Mean steps {:.2f}, Noise {:.2f}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,episodes,np.sum(steps),np.mean(steps),agent.noise_scale,r_mean,r_min,r_max,r_std,(toc-tic)/60))


if __name__ == "__main__":
    algo = "ddpg"
    watch(algo)
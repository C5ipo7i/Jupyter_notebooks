import numpy as np
import time
from collections import deque

from plot import plot
"""
Actions shape: (N,E,F) Where N = Batch size, E = num elevators, F = num floors
"""
def deterministic_actions(probs):
    probs = probs.squeeze(0)
    actions = np.zeros(probs.shape)
    choices = np.arange(probs.shape[-1])
    for elevator in range(actions.shape[0]):
        action_mask = np.where(probs[elevator,:] == np.max(probs[elevator,:]))[0]
        actions[elevator,action_mask] = 1
    return actions


def into_action(probs):
    probs = probs.squeeze(0)

    actions = np.zeros(probs.shape)
    choices = np.arange(probs.shape[-1])
#     print('choices',choices)
    for elevator in range(actions.shape[0]):
        action_mask = np.random.choice(choices,p=probs[elevator,:])
#         print('action_mask',action_mask)
        actions[elevator,action_mask] = 1
#         print('actions',actions)
    return actions

def seed_replay_buffer(env, agent, min_buffer_size):
    printing = False
    obs = env.reset()
    while len(agent.PER) < min_buffer_size:
        # Random actions between 1 and -1
        probs = agent.act(obs)
#         print('agent probs',actions)
        actions = into_action(probs)
#         print('agent action',actions)
        
        next_obs,rewards,dones = env.step(actions,printing)
        # reshape
        agent.add_replay_warmup(obs,probs,rewards,next_obs,dones)
        # Store experience
        if dones:
            obs = env.reset()
        obs = next_obs
    print('finished replay warm up')

def train_ddpg(env, agent, config):
    printing = False
    episodes,tmax = config.episodes,config.tmax
    tic = time.time()
    means = []
    mins = []
    maxes = []
    stds = []
    mean_steps = []
    steps = []
    scores_window = deque(maxlen=100)
    for e in range(1,episodes+1):
        agent.reset_episode()
        agent.scale_noise()
        episode_scores = []
        obs = env.reset()
        for t in range(tmax):
            probs = agent.act(obs)
            actions = into_action(probs)
            next_obs,rewards,dones = env.step(actions,printing)
            # Step agent with reshaped observations
            agent.step(obs, probs, rewards, next_obs, dones)
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
        if e % 5 == 0:
            toc = time.time()
            r_mean = np.mean(scores_window)
            r_max = max(scores_window)
            r_min = min(scores_window)
            r_std = np.std(scores_window)
            agent.save_weights(config.checkpoint_path)
            plot(means,stds,name=config.name)
            print("\rEpisode: {} out of {}, Steps {}, Mean steps {:.2f}, Noise {:.2f}, Rewards: mean {:.2f}, min {:.2f}, max {:.2f}, std {:.2f}, Elapsed {:.2f}".format(e,episodes,np.sum(steps),np.mean(steps),agent.noise_scale,r_mean,r_min,r_max,r_std,(toc-tic)/60))
        # if np.mean(scores_window) > config.winning_condition:
#             print('Env solved!')
            # save scores
#             pickle.dump([means,maxes,mins,mean_steps], open(str(config.name)+'_scores.p', 'wb'))
            # save policy
            # agent.save_weights(config.checkpoint_path)
            # break

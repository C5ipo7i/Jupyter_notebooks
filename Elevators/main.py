import argparse

from hotel import Hotel
from config import Config
from ddpg import DDPG
from train import seed_replay_buffer,train_ddpg
from networks import Actor,Critic

def main(args,algo):
    print(args)
    N_floors = args.floors
    N_elevators = args.elevators
    N_people = args.people
    print('N_floors: {}, N_elevators {}, N_people {}'.format(N_floors,N_elevators,N_people))
    # Load the ENV
    env = Hotel(N_floors,N_elevators,N_people)

    # size of each action
    action_space = env.action_space

    # examine the state space 
    state_space = env.state_space
    print('Size of each action: {}, Size of the state space {}'.format(action_space,state_space))
    
    ddpg_config = Config(algo)

    agent = DDPG(state_space, action_space,Actor,Critic,ddpg_config)
    if args.restore == True:
        agent.load_weights(ddpg_config.checkpoint_path)
    # Fill buffer with random actions up to min buffer size
    seed_replay_buffer(env, agent, ddpg_config.min_buffer_size)
    # Train agent
    train_ddpg(env, agent, ddpg_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization for Elevator agent")

    parser.add_argument('--restore',type=bool,default=False, help="(bool) Restores agent from config checkpoint")
    parser.add_argument('--floors',type=int,default=15, help="determines number of floors")
    parser.add_argument('--elevators',type=int,default=2, help="determines number of elevators")
    parser.add_argument('--people',type=int,default=2, help="determines number of people")
    args = parser.parse_args()
    algo = "ddpg"
    main(args,algo)
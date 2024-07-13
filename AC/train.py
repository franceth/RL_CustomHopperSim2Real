"""Train an RL agent on the OpenAI Gym Hopper environment using
    Actor-critic algorithm
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

from timeit import default_timer as timer
#import wandb

#wandb.login()

#wandb.init(
#		# Set the project where this run will be logged
#		project="RLProjectA-C", # cambiare
#		# We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#		name="train_actor_critic",
#		# Track hyperparameters and run metadata
#		config={
#		"learning_rate": 1e-3,
#		"architecture": "Actor_critic",
#		"dataset": "none",
#		"epochs": 150000,
#		})



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=10000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=200, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)
	
	start_time = timer()
	episode_rewards = []
	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state

		while not done:  # Loop until the episode is over
			action, action_probabilities = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, reward, done) 

			train_reward += reward

		
		agent.update_policy()
		agent.clear_data()
		episode_rewards.append(train_reward)
		#wandb.log({"Train Reward": train_reward, "Episode": episode})

		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)

	end_time = timer()
	print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

	torch.save(agent.policy.state_dict(), "modelAC.mdl")

if __name__ == '__main__':
	main()
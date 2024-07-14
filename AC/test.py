"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy
#import wandb

from timeit import default_timer as timer
#wandb.login()

#wandb.init(
#		# Set the project where this run will be logged
#		project="RLProjectA-C", 
#		# We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
#		name="test_actor_critic",
#		# Track hyperparameters and run metadata
#		config={
#		"learning_rate": 1e-3,
#		"architecture": "Actor_critic",
#		})

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='modelAC.mdl', type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=500, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-target-v0')
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	policy.load_state_dict(torch.load(args.model), strict=True)

	agent = Agent(policy, device=args.device)
	total_reward = 0
	start_time = timer()
	for episode in range(args.episodes):
		done = False
		test_reward = 0
		state = env.reset()

		while not done:

			action, _ = agent.get_action(state, evaluation=True)

			state, reward, done,_ = env.step(action.detach().cpu().numpy())

			if args.render:
				env.render()

			test_reward += reward
		total_reward += test_reward
		#wandb.log({"Test Reward": test_reward, "Episode":episode})
		#print(f"Episode: {episode} | Return: {test_reward}")

	end_time = timer()
	#wandb.finish()
	print(f"Mean reward: {total_reward / args.episodes}")
	print(f"[INFO] Total test time: {end_time-start_time:.3f} seconds")
if __name__ == '__main__':
	main()
"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE algorithm
"""
import argparse
import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy
from timeit import default_timer as timer

#import wandb
#wandb.login()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=15000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=2000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()


def main():

	env = gym.make('CustomHopper-source-v0')
	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())

	"""
		Training
	"""
	#wandb.init(
	#	# Set the project where this run will be logged
	#	project="RLProjectREINFORCE_TRAIN",
	#	# We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
	#	name="baseline",
	#	# Track hyperparameters and run metadata
	#	config={
	#	"learning_rate": 1e-3,
	#	"architecture": "Reinforce",
	#	"dataset": "none",
	#	})


	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	agent = Agent(policy, device=args.device)

	episode_rewards = []
	start_time = timer()
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
		#wandb.log({"Train Reward": train_reward, "Episode": episode,})
		if (episode+1)%args.print_every == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)
	end_time = timer()
	#wandb.finish()
	print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

	torch.save(agent.policy.state_dict(), "modelREINFORCE.mdl")
	

if __name__ == '__main__':
	main()
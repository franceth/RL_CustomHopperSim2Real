from env.custom_hopper import *
from ADR import AutomaticDomainRandomization, ADRCallback
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
wandb.login()
import gym
import argparse
import os

LEARNINING_RATE = 0.000595596286192614
ENTROPY_COEFF = 0.016533686053804394
CLIP_RANGE = 0.3341640546208167
BATH_SIZE = 128

THIGH_MEAN_MASS = 3.92699082
LEG_MEAN_MASS = 2.71433605
FOOT_MEAN_MASS = 5.0893801

sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "mean_reward", "goal": "maximize"},
        "parameters": {
            "delta":{"values": [0.02, 0.05 ,0.1]},
        }
}


def parse_args():
	parser = argparse.ArgumentParser()
	n_envs = os.cpu_count()
	parser.add_argument('--p-b', default=0.50, type=float, help='Probability of testing the Agent in each episode')
	parser.add_argument('--m', default=50, type=int, help='Data buffer size')
	parser.add_argument('--low-th', default=1000, type=int, help='Lower performance threshold')
	parser.add_argument('--high-th', default=1500, type=int, help='Upper performance threshold')
	parser.add_argument('--n-envs', default=n_envs, type=int, help='Number of parallel environments')
	parser.add_argument('--timesteps', default=int(5000000), type=int, help='Timesteps ')
	parser.add_argument('--save-path', default='./models_ADR_TUN/', type=str, help='Path to save models')
	parser.add_argument('--save-freq', default=int(50000), type=int, help='Frequency of saving the model')
	parser.add_argument('--best-model-path', default='./best_models/', type=None, help='Path for the best model found so far')
	return parser.parse_args()

args = parse_args()

global adr_callback
global best_params
best_params={'best_std': 0, 'best_mean' : 0, "best_delta":0}

def objective(envname):
	wandb.init(project="MldlRLproject_Tuning_ADR")
	delta = wandb.config.delta
	timesteps = args.timesteps


	init_params = {"thigh": THIGH_MEAN_MASS,  "leg": LEG_MEAN_MASS, "foot": FOOT_MEAN_MASS}

	handlerADR = AutomaticDomainRandomization(init_params, p_b=args.p_b, m=args.m, delta=delta, thresholds=[args.low_th, args.high_th])
		
	train_env = make_vec_env('CustomHopper-source-v0', n_envs=args.n_envs, vec_env_cls=DummyVecEnv)
	train_env.set_attr(attr_name="bounds", value=handlerADR.get_bounds())
		
	test_env = gym.make('CustomHopper-source-v0')
	eval_callback = EvalCallback(eval_env=test_env, n_eval_episodes=50, eval_freq = args.save_freq, deterministic=True, render=False, best_model_save_path=args.best_model_path+"best_eval_ADR"+str(delta)+"/", warn=False) 
	adr_callback = ADRCallback(handlerADR, train_env, eval_callback, n_envs=args.n_envs, verbose=0, save_freq=args.save_freq, save_path=args.save_path)
	callbacks = CallbackList([adr_callback, eval_callback]) 	
	
	model = PPO('MlpPolicy', train_env, verbose = 0, learning_rate = LEARNINING_RATE, batch_size = BATH_SIZE , ent_coef = ENTROPY_COEFF, clip_range = CLIP_RANGE)
	
	model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)
	mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50, render=False)

	if mean_reward > best_params['best_mean']:
				best_params['best_mean'] = mean_reward
				best_params['best_std'] = std_reward
				best_params['best_delta'] = delta

	wandb.log({"mean_reward": mean_reward, "std_reward":std_reward})
	return mean_reward, std_reward

sweep_id = wandb.sweep(sweep=sweep_configuration, project="MldlRLproject_Tuning_ADR")
wandb.agent(sweep_id, function=lambda:objective("CustomHopper-source-v0"))
print("Best distributions [source]: ",best_params)
wandb.finish()

def main():

    for delta in ["0.02", "0.05", "0.1"]:
        test_env = gym.make('CustomHopper-target-v0')
        test_env = Monitor(test_env)
        model = PPO.load("./best_eval_ADR"+delta+"/best_model")
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
        print(f"[s-t] mean_reward ADR delta ="+delta+":{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
	main()
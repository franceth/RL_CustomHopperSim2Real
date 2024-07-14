import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import argparse
wandb.login()

#best hyperparameters from PPO source Tuning
LEARNINING_RATE = 0.000595596286192614
ENTROPY_COEFF = 0.016533686053804394
CLIP_RANGE = 0.3341640546208167
BATCH_SIZE = 128

THIGH_MEAN_MASS = 3.92699082
LEG_MEAN_MASS = 2.71433605
FOOT_MEAN_MASS = 5.0893801
timestep=0

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--timesteps', default=int(5000000), type=int, help='Timesteps ')
	parser.add_argument('--save-freq', default=int(50000), type=int, help='Frequency of saving the model')
	parser.add_argument('--best-model-path', default='./BestModelUDR', type=None, help='Path for the best model found so far')
	return parser.parse_args()

args = parse_args()


class ResetParametersCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super(ResetParametersCallback, self).__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Check if we are at the start of a new episode
        done_array = self.locals["dones"]
        if any(done_array):
            wandb.log({"reward": self.locals['infos'][0]['episode']['r'], "step": self.num_timesteps} )
            self.reset_parameters()
        return True

    def reset_parameters(self):
        self.env.sample_parameters(delta = wandb.config.delta, thigh_mass = THIGH_MEAN_MASS, leg_mass = LEG_MEAN_MASS, foot_mass = FOOT_MEAN_MASS)

sweep_configuration = {
        "method": "grid",
        "name": "sweep",
        "metric":
            {"name": "mean_reward", "goal": "maximize"},
        "parameters": {
            "delta":{"values":["0.25", "0.50", "0.75"]},
        }
}

global best_params
best_params={'best_std': 0, 'best_mean' : 0, 'delta': ""}

def objective(envname):
    wandb.init(project="MldlRLproject_UDRTuning")

    train_env = gym.make(envname)
    train_env = Monitor(train_env)
    reset_callback = ResetParametersCallback(train_env)
    target_eval_callback = EvalCallback(eval_env = train_env, n_eval_episodes=50, eval_freq=args.save_freq, best_model_save_path=args.best_model_path + wandb.config.delta+"/", verbose = 0)
    callback = CallbackList([target_eval_callback, reset_callback])

    model = PPO("MlpPolicy", train_env, learning_rate = LEARNINING_RATE,
                                        ent_coef = ENTROPY_COEFF, 
                                        clip_range = CLIP_RANGE,
                                        verbose=0)

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    model = PPO.load(args.best_model_path +str(wandb.config.delta)+"/best_model")

    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=50)

    if mean_reward > best_params['best_mean']:
            best_params['best_mean'] = mean_reward
            best_params['best_std'] = std_reward
            best_params['delta'] = str(wandb.config.delta)

    wandb.log({"mean_reward": mean_reward, "std_reward":std_reward})
    return mean_reward, std_reward

sweep_id = wandb.sweep(sweep=sweep_configuration, project="MldlRLproject_UDRTuning")
wandb.agent(sweep_id, function=lambda:objective("CustomHopper-source-v0"))
wandb.finish()

def main():

    for delta in ["0.25", "0.50", "0.75"]:
        test_env = gym.make('CustomHopper-target-v0')
        test_env = Monitor(test_env)
        model = PPO.load(args.best_model_path+delta+"/best_model")
        mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=50, render=False, warn=False)
        print(f"[s-t] mean_reward UDR delta ="+delta+":{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == '__main__':
    main()
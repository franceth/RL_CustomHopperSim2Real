import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import wandb

EPS = 2e-15

class AutomaticDomainRandomization():
        def __init__(self, init_params: dict, p_b = 0.5, m = 50, delta = 0.05, thresholds = [1000, 1600] ) -> None:
            self.init_params = init_params # initial values of each part of the body
            self.thresholds = thresholds
            self.delta = delta
            self.m = m # number of episodes, dimension of buffer
            self.bounds = self._init_bounds()
            self.p_b = p_b
            self.thigh_mass = None
            self.leg_mass = None
            self.foot_mass = None
            self.rewards = []
            self.partOfBody = ['thigh', 'leg', 'foot']
            self.buffer = {
                "thigh_low": [],
                "thigh_high": [],
                "leg_low": [],
                "leg_high": [],
                "foot_low": [],
                "foot_high": []
            }
            self.keys = list(self.buffer.keys())
            self.performances = {
                "thigh_low": [],
                "thigh_high": [],
                "leg_low": [],
                "leg_high": [],
                "foot_low": [],
                "foot_high": []
            }
            self.increments = []

        def _init_bounds(self):
            dict = {
                "thigh_low": self.init_params['thigh']-EPS,
                "thigh_high": self.init_params['thigh']+EPS,
                "leg_low": self.init_params['leg']-EPS,
                "leg_high": self.init_params['leg']+EPS,
                "foot_low": self.init_params['foot']-EPS,
                "foot_high": self.init_params['foot']+EPS
            }
            return dict
        def _constant(self, *args):
            return self.delta
        def compute_entropy(self):
            range_thigh = abs(self.bounds['thigh_high'] - self.bounds['thigh_low'])
            range_leg = abs(self.bounds['leg_high'] - self.bounds['leg_low'])
            range_foot = abs(self.bounds['foot_high'] - self.bounds['foot_low'])
            entropy = np.log([range_thigh, range_leg, range_foot]).mean()
            return entropy
        
        def append_reward_bodypart(self, partOfBody: str, reward: float):
            self.buffer[partOfBody].append(reward)
            return None
        
        def evaluate_performance(self, partOfBody: str): # compute the performance of a single part of body as the mean of all rewards
            performance = np.mean(np.array(self.buffer[partOfBody]))
            self.buffer[partOfBody].clear()
            return performance
        
        def update_ADR(self, partOfBody: str):
            if len(self.buffer[partOfBody]) >= self.m:
                highOrLow = partOfBody.split('_')[1]
                performance = self.evaluate_performance(partOfBody)
                self.performances[partOfBody].append(performance)

                if performance >= self.thresholds[1]:
                    if highOrLow == "high":
                        self.increase(partOfBody)
                    else:
                        self.decrease(partOfBody)
                if performance <= self.thresholds[0]:
                    if highOrLow == "low":
                        self.increase(partOfBody)
                    else:
                        self.decrease(partOfBody)
            wandb.log({"bound" : self.get_bounds()})
            
            return None
        
        def get_bounds(self):
            return self.bounds
            
        def random_masses(self):
            #sampling body masses 
            thighVal = np.random.uniform(self.bounds["thigh_low"], self.bounds["thigh_high"])
            footVal = np.random.uniform(self.bounds["foot_low"], self.bounds["foot_high"])
            legVal = np.random.uniform(self.bounds["leg_low"], self.bounds["leg_high"])

            bodyParts = {"thigh": thighVal, "foot": footVal, "leg": legVal}

            pb = np.random.uniform(0,1)
            randomCompletePart = self.set_random_parameter(bodyParts)
            
            if pb < self.p_b:
                part = randomCompletePart+"_low"
            else:
                part = randomCompletePart+"_high"

            bodyParts[randomCompletePart] = self.bounds[part]
            return list(bodyParts.values()), part

        def evaluate(self, reward, randomCompletePart):
            self.append_reward_bodypart(partOfBody = randomCompletePart, reward = reward)
            self.update_ADR(partOfBody = randomCompletePart)
            return None        
             
        def increase(self, partOfBody: str):
            type = partOfBody.split('_')[0]
            high_bound = type + '_high'
            highOrLow = partOfBody.split('_')[1]
            new_bound = self.bounds[partOfBody] + self.delta
            if (highOrLow == 'low' and new_bound > self.bounds[high_bound]):
                new_bound = new_bound - self.delta
            self.bounds[partOfBody] = new_bound
            return None
        
        def decrease(self, partOfBody: str):
            type = partOfBody.split('_')[0]
            low_bound = type + '_low'
            highOrLow = partOfBody.split('_')[1]
            new_bound = self.bounds[partOfBody] - self.delta
            if (highOrLow == 'high' and new_bound > 0.0 and new_bound < self.bounds[low_bound]):
                new_bound = new_bound + self.delta
            if new_bound <= 0.0:
                new_bound = new_bound + self.delta
            self.bounds[partOfBody] = new_bound
            return None
        
        def set_random_parameter(self, bodyParts):
            keys = list(bodyParts.keys())
            rand = np.random.choice(len(keys))
            name = keys[rand]
            return name
        
class ADRCallback(BaseCallback):
        def __init__(self, handlerADR : AutomaticDomainRandomization, vec_env, eval_callback, n_envs = 1, verbose = 0, save_freq = 1000, save_path:str='./models', name_prefix:str='adr_model'):
            super(ADRCallback, self).__init__(verbose)
            self.adr = handlerADR
            self.n_envs = n_envs
            self.vec_env = vec_env
            self.eval_callback = eval_callback
            self.bounds_used = [None] * n_envs
            self.n_episodes = 0
            self.save_freq = save_freq
            self.save_path = save_path
            self.name_prefix = name_prefix
        
        def _on_step(self):
            for done, infos, nr_env, bound_used in zip(self.locals['dones'], self.locals['infos'], range(self.vec_env.num_envs), self.bounds_used):
                wandb.log({"step": self.num_timesteps, "entropy": self.adr.compute_entropy()})
                if(done):
                    self.n_episodes += 1
                    wandb.log({"reward": infos['episode']['r'], "step": self.num_timesteps})
                    if bound_used is not None:
                        self.adr.evaluate(infos['episode']['r'], bound_used)
                    env_params, self.bounds_used[nr_env] = self.adr.random_masses()
                    self.vec_env.env_method('set_parameters', env_params, indices=nr_env)


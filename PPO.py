import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import minerl
import numpy as np
import logging
from network import ConvNet


logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def converter(observation):
    region_size = 8
    obs = observation['pov']
    obs = obs / 255
    H,W,C = obs.shape
    obs = torch.from_numpy(obs).float().to(device)
    # if len(state.shape) < 4:
    #         state = torch.unsqueeze(state, 0)
    # state = state.flatten()
    obs = obs.reshape((C,H,W))
    compass = observation["compassAngle"]
    compass = torch.tensor(np.array([compass]), dtype=torch.float32).to(device)
    return obs, compass


class Memory:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.compassAngles = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.observations[:]
        del self.compassAngles[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        # self.action_layer = nn.Sequential(
        #         nn.Linear(state_dim, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, action_dim),
        #         nn.Softmax(dim=-1)
        #         )
        # self.action_layer = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(800,action_dim),
        #     nn.Softmax(dim=-1)
        # )
        self.action_layer = ConvNet(64, 64, action_dim, True)
        # critic
        # self.value_layer = nn.Sequential(
        #         nn.Linear(state_dim, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, n_latent_var),
        #         nn.Tanh(),
        #         nn.Linear(n_latent_var, 1)
        #         )
        # self.value_layer = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Flatten(),
        #     nn.Linear(800, 1)
        # )
        self.value_layer = ConvNet(64, 64, 1)

    def forward(self):
        raise NotImplementedError
        
    def act(self, obs, compass, memory):
        # state = torch.from_numpy(state).float().to(device)
        if len(obs.shape) < 4:
            obs = torch.unsqueeze(obs, 0)
        if len(obs.shape) > 4:
            obs = torch.squeeze(obs, 1)
        action_probs = self.action_layer(obs, compass)
        dist = Categorical(action_probs)
        action = dist.sample()

        memory.compassAngles.append(compass)
        memory.observations.append(obs)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, obs, compass, action):
        if len(obs.shape) < 4:
            obs = torch.unsqueeze(obs, 0)
        if len(obs.shape) > 4:
            obs = torch.squeeze(obs, 1)
        action_probs = self.action_layer(obs, compass)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        state_value = self.value_layer(obs, compass)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy
        
class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_observations = torch.stack(memory.observations).to(device).detach()
        old_compass = torch.stack(memory.compassAngles).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_observations, old_compass, old_actions)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
def main():
    ############## Hyperparameters ##############
    env_name = 'MineRLNavigateDense-v0'
    from environment import MyEnv
    # creating environment
    env = MyEnv()
    state_dim = 3* 64* 64
    action_dim = 7
    render = False
    solved_reward = 200         # stop training if avg_reward > solved_reward. this is impossible
    log_interval = 1           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 10000         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 128*8      # update policy every n timesteps
    lr = 0.00025
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    save_interval = 5
    #############################################
    
    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0

    episode_rewards = []

    # training loop
    for i_episode in range(1, max_episodes+1):
        episode_reward = 0
        obs, compass = converter(env.reset())
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old:
            action = ppo.policy_old.act(obs, compass, memory)
            state, reward, done, _ = env.step(action)
            obs, compass = converter(state)
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            episode_reward += reward
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
            running_reward += reward
            if render:
                env.render()
            if done:
                break
            logging.debug(f"instant reward {reward}, timestep {timestep}")
        episode_rewards.append(episode_reward)
        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            logging.info("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            logging.debug('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), './PPO_{}_{}.pth'.format(env_name, i_episode))
            np.save('./PPO_ep_rewards_{}_{}'.format(env_name, i_episode), np.array(episode_rewards))
if __name__ == '__main__':
    main()
    

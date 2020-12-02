import gym
import logging
from PPO import PPO, Memory, converter
from PIL import Image
import torch
import numpy as np

from environment import MyEnv


def test():
    ############## Hyperparameters ##############

    # creating environment
    env = MyEnv()
    env_name = env.env_name
    action_dim = 5
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 100
    max_timesteps = 5000
    save_gif = False

    filename = "PPO_{}_265.pth".format(env_name)
    # directory = "./preTrained/"
    
    memory = Memory()
    ppo = PPO(64*64*3, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(filename))
    rewards = []
    for ep in range(1, n_episodes+1):
        ep_reward = 0
        state = env.reset()
        for t in range(max_timesteps):
            obs, compass = converter(state)
            action = ppo.policy_old.act( obs=obs, compass=compass, memory=memory)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            # if render:
            #     env.render()
            if save_gif:
                 img = obs.data.numpy()
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))  
            if done:
                break
        rewards.append(ep_reward)
        logging.debug('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    np.save('./PPO_ep_rewards_test_{}'.format(env_name), np.array(rewards))
if __name__ == '__main__':
    test()
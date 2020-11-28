import gym
from PPO import PPO, Memory, converter
from PIL import Image
import torch

from environment import MyEnv


def test():
    ############## Hyperparameters ##############

    # creating environment
    env = MyEnv()
    env_name = env.env_name
    # state_dim = env.observation_space.shape[0]
    action_dim = 7
    render = False
    max_timesteps = 500
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    #############################################

    n_episodes = 3
    max_timesteps = 10000
    render = True
    save_gif = False

    filename = "PPO_{}_95.pth".format(env_name)
    # directory = "./preTrained/"
    
    memory = Memory()
    ppo = PPO(64*64*3, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    
    ppo.policy_old.load_state_dict(torch.load(filename))
    
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
            
        print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    
if __name__ == '__main__':
    test()
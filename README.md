# PPO-PyTorch
Minimal PyTorch implementation of Proximal Policy Optimization with clipped objective for OpenAI gym environments.

## Usage

- To test a preTrained network : run `test.py` or `test_continuous.py`
- To train a new network : run `PPO.py`
- All the hyperparameters are in the `PPO.py` file

## Dependencies
Trained and tested on:
```
Python 3.6
PyTorch 1.0
NumPy 1.15.3
gym 0.10.8
Pillow 5.3.0
minerl 0.3.0
```

## Results

PPO MineRLNavigateDense-v0 train1 (120 episodes)           |  PPO MineRLNavigateDense-v0 train2 (4000 episodes)

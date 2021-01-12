# BridgeBidding.jl - Model Training

The reinforcement learning methods are implemented here. (Supervised training is done via vanilla Knet.) All the training is done over a flexible actor-critic iterator. You can change the losses for actors and critics in order to use a particular policy gradient methods.

### Contents

- **actorcritic.jl**
Generic training iterator for actor critic methods. Also contains some shortcuts for REINFORCE, A2C, and PPO.
- **loss.jl**
Collection of different actor or critic loss functions.
- **scheduler.jl**
Generic scheduler structure that keeps track of how many times it has been called and changes the returning value accordingly.
- **opponent/**
Structures for opponent schedulers. These are called with the current version of the model in order to integrate that into the logic.

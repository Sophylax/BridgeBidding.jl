# BridgeBidding.jl - Models

A Bridge model is a callable struct populated with *Knet*'s *Param* structures. They take a bridge representation and output either a policy vector, an estimation of value or action-value vector, or a combination of these three. We apply two abstractions on top of this structure:

- Each model is expected to implement `getobserver(mdl::ModelType)`, which indicates how they expect their data to be encoded. An observer is a structure which turns a vector of *Observation*s into a predefined encoding. With this abstraction, we can share identical encodings between model types and let the rest of the code forget about how to encode the state.

- Each model is also expected to implement at least one of these `getpolicy(mdl::ModelType)`, `getvalue(mdl::ModelType)`, `getactionvalue(mdl::ModelType)` depending on the meaning of their output. (Some abstract types for models that output only one or all of these are predefined.) With this abstraction, the rest of the code doesn't need to keep track of what exactly the network outputs in what order. There is also some simple structure to merge multiple models doing different functions into a single combined model when needed.

### Contents

- **model.jl**
Primitive definitions of BridgeModel type and many variants of it.
- **observer/**
The generic structure that turns Vector of *Observation* into a format the models can accept. Each model will designate an observer that they can work with.
- **rnnalts/**
Collection of different ways of using rnn in the policy models. Experiments show MkIV as the best, which is used for other types of models.
- **primitives.jl**
Collection of neural network primitive structures (embedding layers, fully connected layers, etc).
- **allpass.jl**
Simplest policy model which always returns an always passing policy.
- **puremlp.jl**
Policy model with just a single MLP.
- **rnn[\*].jl**
Different RNN models with different output heads.
- **selfattnmlp.jl**
Model with self attention instad of an RNN. Uses transformer primitives from *transformerparts.jl*.
- **gong.jl**
Model that attempts to replicate Gong et al 2019, along with an Observer that conforms their representation.
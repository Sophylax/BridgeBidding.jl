# BridgeBidding.jl - Model Evaluation

This section is divided into three: Evaltuation Iterators, Supervised Evaluators, Reinforcement Evaluators. All evaluators are initialized with the evaluation datasets and called at evaluation time with the model.

### Evaluation Iterators:
- [evaluate.jl:](evaluate.jl) An iterator wrapper that calls an external function every N iterations. Can either return the value of iterator, the external function, or both.
- [timedevaluate.jl:](timedevaluate.jl) As above, but instead of calling the function every N iterations, it attempts to maintain a ratio of execution time between the inner iterator and the external function.

### Supervised Evaluators
- ***supervised[\*].jl:*** These files all describe a different way of evaluation over a supervised dataset: Evaluating bids, value functions, action-value function, or all three at once.

### Reinforcement Evaluators
- [gamescore.jl:](gamescore.jl) The model plays versus itself and the result is the absolute distance to the par score.
- [adversarial.jl:](adversarial.jl) The model plays versus a different model (given at evaluation time) in a duplicate format, the result is relative performance.
- [noncomp.jl:](noncomp.jl) The model plays versus an always passing model, and result is compared to non-competitive par score.

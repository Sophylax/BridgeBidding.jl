# BridgeBidding.jl - Source Code

The source code is divided into six folders, each functioning around a single central structure:

* [Dataset Management:](data) implements dataset loaders and iterators.
* [Bridge Environment:](env) implements the mechanics of the Bridge Bidding environment.
* [Game Episodes:](episode) implements structured ways of playing Bridge Bidding with the given models.
* [Evaluation:](episode) implements the evaluation metrics.
* [Models:](episode) implements various different models that can be used to play Bridge.
* [Train:](episode) implements the reinforcement learning methods to train the models.
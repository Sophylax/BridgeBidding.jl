# BridgeBidding.jl - Bridge Episode Playing

Logic for playing whole episodes of bridge bidding games. This folder mainly works with the abstract type *GameFormat*. It declares a way of playing the game and is expected to be callable with the following signature:
```julia
(gFormat::GameFormat)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
```
This function is expected to play tables initialized with the given double dummy instances using the two *PolicyModel* (or equivalent) given as the partnerships. It should return an array of *Experience* collected from actor and/or opponent. It also should respect the greedy or stochastic ways of deciding on an action, and passes the nonzeropass to the newly generated *BridgeState* instances.

### Contents

- [game.jl:](game.jl) Declares *GameFormat* and has the previous way of playing the episodes. Some code might still refer to it, but it's functions are deprecated.
- [singlegame.jl:](singlegame.jl) Declares *SingeGame* format, in which the tables are played once and optionally the scores are compared to the par score.
- [duplicategame.jl:](duplicategame.jl) Declares *DuplicateGame* format, in which the tables are played twice and the scores are compared to eachother.
- [passtrickduplicategame.jl:](passtrickduplicategame.jl) Variation over *DuplicateGame*, in which the reward symmetry is broken if both tables all pass.
- [passtrickduplicategame_mk2.jl:](passtrickduplicategame_mk2.jl) Variation over *DuplicateGame*, in which the reward symmetry is broken if any table all pass.
- [experience.jl:](experience.jl) Declares the *Experience* structure, representing a single point of information generated during gameplay. Also declares *ExperienceBatch*, a way to group experiences and access the batched versions of some fields.
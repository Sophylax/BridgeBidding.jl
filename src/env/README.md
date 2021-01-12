# BridgeBidding.jl - Bridge Environment

Logic for defining and interacting with a bridge bidding environment.

### Contents

- [state.jl:](state.jl) The main structure, defining BridgeState and many functions that interacts with it.
- [observation.jl:](observation.jl) Defines Observation, our standard way of representing a player's information about the state.
- [score.jl:](score.jl) Auxilliary logic about determining the normal or par score.
- [types.jl:](types.jl) Primitive definitions about card, bid, and player enumeration.
- [dda.jl:](dda.jl) On-demand functions that call an external GIB engine to solve doubledummy for a game state. Mostly unused since changing ways to interact with Wbridge5.
- [pbn.jl:](pbn.jl) Functions to load PBN files into our BridgeState. Very primitive and mostly unusued since changing ways to interact with Wbridge5.
- [dd.sh:](dd.sh) Bash script that *dda.jl* uses to interact with GIB.
# BridgeBidding.jl - Data Loaders

Logic for loading the datasets we have and construct iterators around them. Each dataset is an iterator that will batch a (random) selection of data and return them.

### Contents

- [doubledummy.jl:](doubledummy.jl) Logic to work with the Double Dummy games for RL.
- [supervisedbid.jl:](supervisedbid.jl) Logic to work with the supervised games for the bid classification task.
- [supervisedmulti.jl:](supervisedmulti.jl) Logic to work with the supervised games for the any task (bid, value, action-value tasks).

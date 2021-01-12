# BridgeBidding.jl - Learning Bridge Bidding using Knet

Source code and datasets for my masters thesis: "Training a Bridge Bidding Agent using Minimal Feature Engineering and Deep Reinforcement Learning". It contains various ways to apply Reinforcement Learning with Policy Methods to the Bridge Bidding problem using Julia and Knet.

### Repository Structure
* [Source:](src) implementation of everything for the bridge learning process.
* [Datasets:](data) collection of various datasets utilized for either supervised or reinforcement learning.
* [Scripts:](srcript) collection of handful script that I've used that can either be directly run or serve as examples.

## Getting Started

### Prerequisites

Following Julia packages are required for this repository.
```julia
using Pkg; Pkg.add(["Knet", "JSON", "Random", "StatsBase", "IterTools"])
```

### Example Script

```julia
#Load BridgeBidding.jl and Knet
include("BridgeBidding/src/BridgeBidding.jl"); using .BridgeBidding, Knet;

#Load the datasets
ddtrn = generate_doubledummy_gameset("BridgeBidding/data/dd-train-5120k.txt", batchsize=64)[1];
ddevl = generate_doubledummy_gameset("BridgeBidding/data/dd-eval-10k.txt", batchsize=64)[1];

#Define the model and it's optimizers
model = RnnMlpMultiHeadModel(cardembed=64,bidembed=64,vulembed=64,lstmhidden=1024,mlphidden=[1024]);
for param in Knet.params(model); param.opt = Rmsprop(lr=1e-3); end;

#Set-up a standard Advantage Actor Critic process
trainer = A2C(model, ddtrn);

#Add an evaluation loop every 800 iterations
gse = GameScoreEvaluator(ddevl);
evalfun = ()->begin println(stderr); gse(model); end
trainer = evaluate(evalfun, trainer, cycle=800);

#Run the Reinforcement Learning process
progress!(trainer);
```
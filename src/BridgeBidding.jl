module BridgeBidding

using Knet, JSON, Random, StatsBase, IterTools

atype=gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

include("env/state.jl"); export BridgeState
include("env/observation.jl")
include("env/score.jl")
include("env/types.jl")
include("env/dda.jl")
include("env/pbn.jl"); export BridgeStatesFromPBN

include("data/supervisedbid.jl"); export SupervisedBidSet, generate_supervised_bidset
include("data/supervisedmulti.jl"); export SupervisedMultiSet, generate_supervised_multiset
include("data/doubledummy.jl"); export DoubleDummySet, generate_doubledummy_gameset

include("episode/experience.jl")
include("episode/game.jl"); export playepisode!, playepisode
include("episode/singlegame.jl"); export SingleGame
include("episode/duplicategame.jl"); export DuplicateGame
include("episode/passtrickduplicategame.jl"); export PassTrickDuplicateGame
include("episode/passtrickduplicategame_mk2.jl"); export PassTrickDuplicateGameMkII

include("model/observer/observer.jl")
include("model/observer/variablehistory.jl")
include("model/observer/fixedhistory.jl")
include("model/observer/maskedhistory.jl")
include("model/observer/splithistory.jl")

include("model/model.jl"); export BridgeModel, BridgeMergedModel
include("model/primitives.jl");
#include("model/transformerparts.jl");
include("model/puremlp.jl"); export MlpModel, DropoutMlpModel

include("model/rnnalts/mk1.jl"); export RnnMlpModel, DropoutRnnMlpModel
include("model/rnnalts/mk2.jl"); export RnnMlpModelMkII, DropoutRnnMlpModelMkII
include("model/rnnalts/mk3.jl"); export RnnMlpModelMkIII, DropoutRnnMlpModelMkIII
include("model/rnnalts/mk4.jl"); export RnnMlpModelMkIV, DropoutRnnMlpModelMkIV
include("model/rnnalts/mk5.jl"); export RnnMlpModelMkV, DropoutRnnMlpModelMkV

include("model/gong.jl"); export GongModel

include("model/rnnmulti.jl"); export RnnMlpMultiHeadModel, DropoutRnnMlpMultiHeadModel
include("model/rnnvalue.jl"); export RnnMlpValueModel, DropoutRnnMlpValueModel
include("model/rnnactionvalue.jl"); export RnnMlpActionValueModel, DropoutRnnMlpActionValueModel
include("model/allpass.jl"); export AllPassModel

#include("model/selfattnmlp.jl"); export SelfAttnMlpModel

include("train/scheduler.jl"); export AnyScheduler

include("train/opponent/scheduler.jl")
include("train/opponent/self.jl"); export SelfScheduler
include("train/opponent/fixed.jl"); export FixedScheduler
include("train/opponent/dual.jl"); export DualScheduler
include("train/opponent/staggered.jl"); export StaggeredScheduler
include("train/opponent/zoo.jl"); export ZooScheduler

include("train/loss.jl"); export ActorLoss, ValuePrediction, FinalMinusValue, OneStepValueDifference, MonteCarloLoss, TDZeroLoss, ProximalLoss, PositiveContractBidAmplifier
#include("train/reinforce.jl"); export Reinforce, Reinforce!
include("train/actorcritic.jl"); export ActorCritic, ActorCritic!, A2C, A2C!, ProximalPolicyOptimization, ProximalPolicyOptimization!, REINFORCE, REINFORCE!
#include("train/actorcriticduplicate.jl"); export ActorCriticDuplicate, ActorCriticDuplicate!
#include("train/proximal.jl"); export ProximalPolicyOptimization, ProximalPolicyOptimization!

include("eval/evaluate.jl"); export evaluate
include("eval/timedevaluate.jl"); export timedevaluate
include("eval/supervisedbid.jl"); export SupervisedBidEvaluator
include("eval/supervisedmulti.jl"); export SupervisedMultiEvaluator
include("eval/supervisedvalue.jl"); export SupervisedValueEvaluator
include("eval/supervisedactionvalue.jl"); export SupervisedActionValueEvaluator
include("eval/gamescore.jl"); export GameScoreEvaluator
include("eval/adversarial.jl"); export AdversarialEvaluator
include("eval/noncomp.jl"); export NonCompEvaluator

end # module

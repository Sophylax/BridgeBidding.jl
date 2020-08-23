module BridgeBidding

using Knet, JSON, Random, StatsBase

atype=gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

include("env/state.jl"); export BridgeState
include("env/observation.jl")
include("env/oracleobservation.jl")
include("env/score.jl")
include("env/types.jl")

include("data/supervisedbid.jl"); export SupervisedBidSet, generate_supervised_bidset
include("data/supervisedmulti.jl"); export SupervisedMultiSet, generate_supervised_multiset
include("data/doubledummy.jl"); export DoubleDummySet, generate_doubledummy_gameset

include("episode/game.jl"); export playepisode!, playepisode

include("model/observer/observer.jl")
include("model/observer/variablehistory.jl")
include("model/observer/fixedhistory.jl")
include("model/observer/maskedhistory.jl")
include("model/observer/splithistory.jl")

#include("model/controller/controller.jl")

include("model/model.jl"); export BridgeModel
include("model/primitives.jl");
#include("model/transformerparts.jl");
include("model/puremlp.jl"); export MlpModel, DropoutMlpModel

include("model/rnnalts/mk1.jl"); export RnnMlpModel, DropoutRnnMlpModel
include("model/rnnalts/mk2.jl"); export RnnMlpModelMkII, DropoutRnnMlpModelMkII
include("model/rnnalts/mk3.jl"); export RnnMlpModelMkIII, DropoutRnnMlpModelMkIII
include("model/rnnalts/mk4.jl"); export RnnMlpModelMkIV, DropoutRnnMlpModelMkIV
include("model/rnnalts/mk5.jl"); export RnnMlpModelMkV, DropoutRnnMlpModelMkV

include("model/rnnmulti.jl"); export RnnMlpMultiHeadModel, DropoutRnnMlpMultiHeadModel
include("model/rnnvalue.jl"); export RnnMlpValueModel, DropoutRnnMlpValueModel
include("model/rnnactionvalue.jl"); export RnnMlpActionValueModel, DropoutRnnMlpActionValueModel

#include("model/selfattnmlp.jl"); export SelfAttnMlpModel

include("train/reinforce.jl"); export Reinforce, Reinforce!
include("train/actorcritic.jl"); export ActorCritic, ActorCritic!

include("eval/evaluate.jl"); export evaluate
include("eval/timedevaluate.jl"); export timedevaluate
include("eval/supervisedbid.jl"); export SupervisedBidEvaluator
include("eval/supervisedmulti.jl"); export SupervisedMultiEvaluator
include("eval/supervisedvalue.jl"); export SupervisedValueEvaluator
include("eval/supervisedactionvalue.jl"); export SupervisedActionValueEvaluator
include("eval/gamescore.jl"); export GameScoreEvaluator
include("eval/adversarial.jl"); export AdversarialEvaluator

end # module

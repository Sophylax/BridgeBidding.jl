# BridgeModel is the parent type of all of our models. It is split into 5 types.
# - BridgePolicyModel: Which outputs a policy
# - BridgeValueModel: Which outputs a value for the state
# - BridgeActionValueModel: Which outputs a value for each action
# - BridgeMultiModel: Which outputs a all of the above
# - BridgeMergedModel: Isn't itself a model, but a struct merges non-multi models into a single model
# We did this in order to abstract away which kind of outputs a model gives us.
# As an example: Gameplay uses getpolicy function, which works for Policy, Multi, and Merged models without changing a thing.

abstract type BridgeModel end

function getobserver(model::BridgeModel)::Observer
	throw(MethodError(getobserver, (model,)))
end

# model.observer = getobserver(model)
# Feel like I should do model.getpolicy(x), etc.
function Base.getproperty(model::BridgeModel, v::Symbol)
    if v == :observer
        return getobserver(model)
    else
        return getfield(model, v)
    end
end

(model::BridgeModel)(x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))

abstract type BridgePolicyModel <: BridgeModel end

getpolicy(model::BridgePolicyModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))

abstract type BridgeValueModel <: BridgeModel end

getvalue(model::BridgeValueModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))

abstract type BridgeActionValueModel <: BridgeModel end

getactionvalue(model::BridgeActionValueModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))

#Policy, Action Value, Value
abstract type BridgeMultiModel <: BridgeModel end

getpolicy(model::BridgeMultiModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))[1]
getactionvalue(model::BridgeMultiModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))[2]
getvalue(model::BridgeMultiModel, x::Union{Observation, Array{Observation,1}}) = model(model.observer(x))[3]

struct BridgeMergedModel
	policy::Union{BridgePolicyModel, Nothing}
	actionvalue::Union{BridgeActionValueModel, Nothing}
	value::Union{BridgeValueModel, Nothing}
end

function BridgeMergedModel(; policy = nothing, actionvalue = nothing, value = nothing)
    BridgeMergedModel(policy, actionvalue, value)
end

getpolicy(model::BridgeMergedModel, x::Union{Observation, Array{Observation,1}}) = getpolicy(model.policy, x)
getactionvalue(model::BridgeMergedModel, x::Union{Observation, Array{Observation,1}}) = getactionvalue(model.actionvalue, x)
getvalue(model::BridgeMergedModel, x::Union{Observation, Array{Observation,1}}) = getvalue(model.value, x)
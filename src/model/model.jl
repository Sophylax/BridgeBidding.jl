abstract type BridgeModel end

function getobserver(model::BridgeModel)::Observer
	throw(MethodError(getobserver, (model,)))
end
getcontroller(model::BridgeModel) = throw(MethodError(getcontroller, (model,)))

function Base.getproperty(model::BridgeModel, v::Symbol)
    if v == :observer
        return getobserver(model)
    elseif v == :controller
        return getcontroller(model)
    else
        return getfield(model, v)
    end
end

(model::BridgeModel)(x::Observation) = model(model.observer(x))
(model::BridgeModel)(x::Observation, y...) = model(model.observer(x), y...)
(model::BridgeModel)(x::Array{Observation,1}) = model(model.observer(x))
(model::BridgeModel)(x::Array{Observation,1}, y...) = model(model.observer(x), y...)
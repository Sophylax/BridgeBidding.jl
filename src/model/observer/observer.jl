# An observer is a struct that transforms an Observation into the specific representation the models may require.
# We abstracted this because many models require same representations and now we can just point a common observer for that purpose.

abstract type Observer end

observe(obsrvr::Observer, x::Array{Observation,1}) = throw(MethodError(observe, (obsrvr,x)))
observe(obsrvr::Observer, x::Observation) = observe(obsrvr,[x])

(obsrvr::Observer)(x::Array{Observation,1}) = observe(obsrvr,x)
(obsrvr::Observer)(x::Observation) = observe(obsrvr,x)
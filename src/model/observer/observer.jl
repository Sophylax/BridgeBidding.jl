abstract type Observer end

observe(obsrvr::Observer, x::Array{Observation,1}) = throw(MethodError(observe, (obsrvr,x)))
observe(obsrvr::Observer, x::Observation) = observe(obsrvr,[x])

(obsrvr::Observer)(x::Array{Observation,1}) = observe(obsrvr,x)
(obsrvr::Observer)(x::Observation) = observe(obsrvr,x)
mutable struct TimedEvaluate{I}
	func
	iter::I
    start::Bool
    lastdur
    lasttime
	value
    bypass
    ratio
end

timedevaluate(func::Base.Callable, iter::I; bypass=nothing, ratio=10) where {I} = TimedEvaluate{I}(func,iter,false,0,0,nothing,bypass,ratio)
timedevaluate(iter; bypass=nothing, ratio=10)=timedevaluate(()->"",iter, bypass=bypass, ratio=ratio)
Base.length(e::TimedEvaluate) = length(e.iter)

function Base.iterate(e::TimedEvaluate, state=(length(e),))
	if !e.start
        e.start = true
		e.value, e.lastdur, _ = @timed e.func()
        e.lasttime = time()
	end
	remain, inner_state = first(state), Base.tail(state)
    inner_iter = iterate(e.iter, inner_state...)
    if isnothing(inner_iter)
    	return nothing
    end
    if (time()-e.lasttime)/e.ratio > e.lastdur
        e.value, e.lastdur, _ = @timed e.func()
        e.lasttime = time()
    end
    inner_value, inner_state = inner_iter

    if isnothing(e.bypass)
        (inner_value, e.value), (remain-1, inner_state)
    elseif e.bypass
        inner_value, (remain-1, inner_state)
    else
        e.value, (remain-1, inner_state)
    end
end
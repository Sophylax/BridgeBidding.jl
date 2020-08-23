mutable struct Evaluate{I}
	func
	iter::I
    start::Bool
	value
    bypass
end

evaluate(func::Base.Callable, iter::I; bypass=nothing) where {I} = Evaluate{I}(func,iter,false,nothing, bypass)
evaluate(iter; bypass=nothing)=evaluate(()->"",iter, bypass=bypass)
Base.length(e::Evaluate) = length(e.iter)

function Base.iterate(e::Evaluate, state=(length(e),))
	if !e.start
        e.start = true
		e.value = e.func()
	end
	remain, inner_state = first(state), Base.tail(state)
    inner_iter = iterate(e.iter, inner_state...)
    if isnothing(inner_iter)
    	return nothing
    end
    if remain == 1
    	e.value = e.func()
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
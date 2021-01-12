mutable struct AnyScheduler
	sch#::Vector{Tuple{Int,<:Any}}
	iteration::Int
	currentvalue::Any
end

AnyScheduler(sch) = AnyScheduler(sch, 1, nothing)

function (asch::AnyScheduler)()
	queryresult = filter(asch.sch) do (i, v) i == asch.iteration end
	if length(queryresult) > 0
		asch.currentvalue = queryresult[1][2]
	end

    asch.iteration += 1
    asch.currentvalue
end
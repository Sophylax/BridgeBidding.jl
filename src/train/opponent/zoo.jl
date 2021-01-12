mutable struct ZooScheduler <: OpponentScheduler
	modelzoo
	iteration
	period
	self
end

ZooScheduler(period; self=false) = ZooScheduler([], 1, period, self)

function getopponent(schdlr::ZooScheduler, model)
	if schdlr.iteration == 1
		push!(schdlr.modelzoo, deepcopy(model))
		schdlr.iteration = max(schdlr.period, 1)
	else
		schdlr.iteration -= 1
	end

    if schdlr.self
    	return rand([schdlr.modelzoo; model])
    else
    	return rand(schdlr.modelzoo)
    end
end
mutable struct StaggeredScheduler <: OpponentScheduler
	period
	iteration
	staggeredmodel
end

StaggeredScheduler(period) = StaggeredScheduler(period, 1, nothing)

function getopponent(schdlr::StaggeredScheduler, model)
	if schdlr.iteration == 1
		schdlr.staggeredmodel = model
		schdlr.iteration = max(schdlr.period, 1)
	else
		schdlr.iteration -= 1
	end

    return schdlr.staggeredmodel
end
struct FixedScheduler <: OpponentScheduler
	fixedmodel
end

function getopponent(schdlr::FixedScheduler, model)
    return schdlr.fixedmodel
end
struct DualScheduler <: OpponentScheduler
	fixedmodel
end

function getopponent(schdlr::DualScheduler, model)
	if rand(Bool)
    	return schdlr.fixedmodel
    else
    	return model
    end
end
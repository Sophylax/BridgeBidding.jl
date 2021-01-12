struct SelfScheduler <: OpponentScheduler
end

function getopponent(schdlr::SelfScheduler, model)
    return model
end
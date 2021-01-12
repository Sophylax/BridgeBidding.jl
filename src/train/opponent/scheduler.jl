abstract type OpponentScheduler end

getopponent(schdlr::OpponentScheduler, model) = throw(MethodError(getopponent, (schdlr,model)))

(schdlr::OpponentScheduler)(model) = getopponent(schdlr,model)
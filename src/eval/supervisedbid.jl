mutable struct SupervisedBidEvaluator
    trainingset
    testingset
    lastbest
    onnewbest
end

function SupervisedBidEvaluator(trainingset, testingset; onnewbest = nothing)
    SupervisedBidEvaluator(trainingset, testingset, nothing, onnewbest)
end

function (se::SupervisedBidEvaluator)(model)
    println(stderr)
    trn_acc = accuracy(model,se.trainingset)
    @info "Training Set accuracy: $(100*trn_acc)%"
    tst_acc = accuracy(model,se.testingset)
    @info "Test Set accuracy: $(100*tst_acc)%"

    if !isnothing(se.onnewbest)
        if isnothing(se.lastbest)
            se.lastbest = tst_acc
        elseif se.lastbest < tst_acc
            se.lastbest = tst_acc
            se.onnewbest()
        end
    end
    (:trn_acc, trn_acc, :tst_acc, tst_acc)
end
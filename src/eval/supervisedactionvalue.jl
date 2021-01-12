"""
    SupervisedActionValueEvaluator(trainingset, testingset; onnewbest = nothing)

Evaluator that calculates the distance of predicted action-values, when given supervised datasets.

Optionally with *onnewbest*, it will track the best performance and call *onnewbest()* when a new best in test set is reached.

    (sme::SupervisedActionValueEvaluator)(model)

Run the evaluation using the given model.
"""
mutable struct SupervisedActionValueEvaluator
    trainingset
    testingset
    lastbest
    onnewbest
end

function SupervisedActionValueEvaluator(trainingset, testingset; onnewbest = nothing)
    SupervisedActionValueEvaluator(trainingset, testingset, nothing, onnewbest)
end

function evaluateactionvalueonset(model,set)
    cnt = qloss = 0
    for (x,b,v) in set
        qpred = model(x)
        cnt += length(x)
        qloss += sum(abs, qpred[b,:][1:size(qpred,2)+1:end] .- v)
    end
    qloss / cnt
end

function (sme::SupervisedActionValueEvaluator)(model)
    println(stderr)
    trn_q = evaluateactionvalueonset(model,sme.trainingset)
    @info "Training Set Q error: $(trn_q)"
    tst_q = evaluateactionvalueonset(model,sme.testingset)
    @info "Test Set Q error: $(tst_q)"

    if !isnothing(sme.onnewbest)
        if isnothing(sme.lastbest)
            sme.lastbest = tst_q
        elseif sme.lastbest > tst_q
            sme.lastbest = tst_q
            sme.onnewbest()
        end
    end
    (:trn_q, trn_q, :tst_q, tst_q)
end
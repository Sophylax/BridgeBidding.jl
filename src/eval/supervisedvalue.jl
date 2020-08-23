mutable struct SupervisedValueEvaluator
    trainingset
    testingset
    lastbest
    onnewbest
end

function SupervisedValueEvaluator(trainingset, testingset; onnewbest = nothing)
    SupervisedValueEvaluator(trainingset, testingset, nothing, onnewbest)
end

function evaluatevalueonset(model,set)
    cnt = vloss = 0
    for (x,b,v) in set
        vgold = atype(v)
        vpred = model(x)
        cnt += length(x)
        vloss += sum(abs, vpred .- vgold)
    end
    vloss / cnt
end

function (sme::SupervisedValueEvaluator)(model)
    println(stderr)
    trn_v = evaluatevalueonset(model,sme.trainingset)
    @info "Training Set V error: $(trn_v)"
    tst_v = evaluatevalueonset(model,sme.testingset)
    @info "Test Set V error: $(tst_v)"

    if !isnothing(sme.onnewbest)
        if isnothing(sme.lastbest)
            sme.lastbest = tst_v
        elseif sme.lastbest > tst_v
            sme.lastbest = tst_v
            sme.onnewbest()
        end
    end
    (:trn_v, trn_v, :tst_v, tst_v)
end
mutable struct SupervisedMultiEvaluator
    trainingset
    testingset
    lastbest
    onnewbest
end

function SupervisedMultiEvaluator(trainingset, testingset; onnewbest = nothing)
    SupervisedMultiEvaluator(trainingset, testingset, nothing, onnewbest)
end

function multievaluateonset(model,set)
    cor = cnt = qloss = vloss = 0
    for (x,b,v) in set
        v = atype(v)
        bids, qs, vs = model(x)
        (z,n) = accuracy(bids, b; average=false)
        cor += z
        cnt += n
        qloss += sum(abs, qs[b,:][1:size(qs,2)+1:end] .- v)
        vloss += sum(abs, vs .- v)
    end
    cor / cnt, qloss / cnt, vloss / cnt
end

function (sme::SupervisedMultiEvaluator)(model)
    println(stderr)
    trn_acc, trn_q, trn_v = multievaluateonset(model,sme.trainingset)
    @info "Training Set accuracy: $(100*trn_acc)%"
    @info "Training Set Q error: $(trn_q)"
    @info "Training Set V error: $(trn_v)"
    tst_acc, tst_q, tst_v = multievaluateonset(model,sme.testingset)
    @info "Test Set accuracy: $(100*tst_acc)%"
    @info "Test Set Q error: $(tst_q)"
    @info "Test Set V error: $(tst_v)"

    if !isnothing(sme.onnewbest)
        if isnothing(sme.lastbest)
            sme.lastbest = tst_acc
        elseif sme.lastbest < tst_acc
            sme.lastbest = tst_acc
            sme.onnewbest()
        end
    end
    (:trn_acc, trn_acc, :trn_q, trn_q, :trn_v, trn_v,
     :tst_acc, tst_acc, :tst_q, tst_q, :tst_v, tst_v)
end
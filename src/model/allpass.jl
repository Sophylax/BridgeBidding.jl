struct AllPassModel <: BridgePolicyModel end

getobserver(mdl::AllPassModel) = VariableHistoryObserver()

function (mdl::AllPassModel)(hand, past, vul, batchsizes)
    rv = zeros(NUMBIDS, size(vul, 1))
    rv[PASS,:] .= 1e9
    atype(rv)
end
(mdl::AllPassModel)(x::Tuple) = mdl(x...)
(mdl::AllPassModel)(x::Tuple,y) = 0

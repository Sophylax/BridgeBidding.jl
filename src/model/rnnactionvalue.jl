struct RnnMlpActionValueModel <: BridgeModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpActionValueModel) = VariableHistoryObserver()

function (mdl::RnnMlpActionValueModel)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    m_embed = _mk4embedmerger(h_embed, v_embed, p_embed, batchsizes)
    p_hidds = mdl.pastencoder(m_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    tanh.(mdl.dense(p_encod))
end
(mdl::RnnMlpActionValueModel)(x::Tuple) = mdl(x...)
#(mdl::RnnMlpActionValueModel)(x::Tuple,y) = nll(mdl(x),y)
function (mdl::RnnMlpActionValueModel)(x::Tuple, bid_gold, value_gold)
    #value_gold = atype(value_gold)
    actval = mdl(x)
    mean(abs2, actval[bid_gold,:][1:size(actval,2)+1:end] .- value_gold)
end



function RnnMlpActionValueModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpActionValueModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,NUMBIDS)
    )
end

function DropoutRnnMlpActionValueModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpActionValueModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,NUMBIDS), mlpdropout)
    )
end
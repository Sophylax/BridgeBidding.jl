struct RnnMlpValueModel <: BridgeModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpValueModel) = VariableHistoryObserver()

function (mdl::RnnMlpValueModel)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    m_embed = _mk4embedmerger(h_embed, v_embed, p_embed, batchsizes)
    p_hidds = mdl.pastencoder(m_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    tanh.(mdl.dense(p_encod)[1,:])
end
(mdl::RnnMlpValueModel)(x::Tuple) = mdl(x...)
#(mdl::RnnMlpValueModel)(x::Tuple,y) = nll(mdl(x),y)
function (mdl::RnnMlpValueModel)(x::Tuple, bid_gold, value_gold)
    value_pred = mdl(x) 

    mean(abs2, value_pred .- value_gold)
end


function RnnMlpValueModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpValueModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,1)
    )
end

function DropoutRnnMlpValueModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpValueModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,1), mlpdropout)
    )
end
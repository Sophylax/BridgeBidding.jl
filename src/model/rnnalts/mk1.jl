struct RnnMlpModel <: BridgePolicyModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpModel) = VariableHistoryObserver()

function (mdl::RnnMlpModel)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    p_embed = mdl.bidembed(past)
    p_hidds = mdl.pastencoder(p_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    v_embed = mdl.vulembed(vul)
    total_encoding = cat(h_embed,p_encod,v_embed,dims=1)
    mdl.dense(total_encoding)
end
(mdl::RnnMlpModel)(x::Tuple) = mdl(x...)
(mdl::RnnMlpModel)(x::Tuple,y) = nll(mdl(x),y)
function RnnMlpModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed, lstmhidden, numLayers=lstmlayers),
        MLP(cardembed+vulembed+lstmhidden,mlphidden...,NUMBIDS)
    )
end

function DropoutRnnMlpModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(cardembed+vulembed+lstmhidden,mlphidden...,NUMBIDS), mlpdropout)
    )
end
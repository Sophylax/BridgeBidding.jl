#RNNALT MkII - Hand/Vul into hidden layer
struct RnnMlpModelMkII <: BridgePolicyModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    hiddeninit::Linear
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpModelMkII) = VariableHistoryObserver()

function (mdl::RnnMlpModelMkII)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    rnn_hid = mdl.hiddeninit(cat(h_embed,v_embed,dims=1))
    mdl.pastencoder.h = rnn_hid
    p_hidds = mdl.pastencoder(p_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    mdl.dense(p_encod)
end
(mdl::RnnMlpModelMkII)(x::Tuple) = mdl(x...)
(mdl::RnnMlpModelMkII)(x::Tuple,y) = nll(mdl(x),y)
function RnnMlpModelMkII(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpModelMkII(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        Linear(input=cardembed+vulembed,output=lstmhidden),
        RNN(bidembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,NUMBIDS)
    )
end

function DropoutRnnMlpModelMkII(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpModelMkII(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        Linear(input=cardembed+vulembed,output=lstmhidden),
        RNN(bidembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,NUMBIDS), mlpdropout)
    )
end
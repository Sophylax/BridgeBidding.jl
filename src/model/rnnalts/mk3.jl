#RNNALT MkIII - Hand/Vul as first token
struct RnnMlpModelMkIII <: BridgeModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    auxtoken::Linear
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpModelMkIII) = VariableHistoryObserver()

function (mdl::RnnMlpModelMkIII)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    aux_tok = mdl.auxtoken(cat(h_embed,v_embed,dims=1))
    p_embed = cat(aux_tok, p_embed[:,size(aux_tok,2)+1:end], dims=2)
    p_hidds = mdl.pastencoder(p_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    mdl.dense(p_encod)
end
(mdl::RnnMlpModelMkIII)(x::Tuple) = mdl(x...)
(mdl::RnnMlpModelMkIII)(x::Tuple,y) = nll(mdl(x),y)
function RnnMlpModelMkIII(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpModelMkIII(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        Linear(input=cardembed+vulembed,output=bidembed),
        RNN(bidembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,NUMBIDS)
    )
end

function DropoutRnnMlpModelMkIII(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpModelMkIII(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        Linear(input=cardembed+vulembed,output=bidembed),
        RNN(bidembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,NUMBIDS), mlpdropout)
    )
end
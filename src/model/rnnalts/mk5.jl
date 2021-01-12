#RNNALT MkV - Hand/Vul concatenated on all tokens, bids are processed 4 by 4
struct RnnMlpModelMkV <: BridgePolicyModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpModelMkV) = SplitHistoryObserver()

function _mk5embedmerger(h_embed, v_embed, p_embed, batchsizes)
    mega_h = h_embed
    mega_v = v_embed
    for bs in batchsizes[2:end]
        mega_h = cat(mega_h, h_embed[:,1:bs], dims=2)
        mega_v = cat(mega_v, v_embed[:,1:bs], dims=2)
    end
    cat(p_embed, mega_h, mega_v, dims=1)
end

function (mdl::RnnMlpModelMkV)(hand, pasts, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embeds = [mdl.bidembed(past) for past in pasts]
    p_embed = cat(p_embeds..., dims=1)
    m_embed = _mk5embedmerger(h_embed, v_embed, p_embed, batchsizes)
    p_hidds = mdl.pastencoder(m_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    mdl.dense(p_encod)
end
(mdl::RnnMlpModelMkV)(x::Tuple) = mdl(x...)
(mdl::RnnMlpModelMkV)(x::Tuple,y) = nll(mdl(x),y)
function RnnMlpModelMkV(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpModelMkV(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN((4*bidembed)+cardembed+vulembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,NUMBIDS)
    )
end

function DropoutRnnMlpModelMkV(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpModelMkV(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN((4*bidembed)+cardembed+vulembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,NUMBIDS), mlpdropout)
    )
end
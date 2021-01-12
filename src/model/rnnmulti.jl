struct RnnMlpMultiHeadModel <: BridgeMultiModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    pastencoder::RNN
    dense::Chain
end

getobserver(mdl::RnnMlpMultiHeadModel) = VariableHistoryObserver()

function (mdl::RnnMlpMultiHeadModel)(hand, past, vul, batchsizes)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    m_embed = _mk4embedmerger(h_embed, v_embed, p_embed, batchsizes)
    p_hidds = mdl.pastencoder(m_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices_for_rnn(batchsizes)]
    combined = mdl.dense(p_encod)
    combined[1:NUMBIDS,:], tanh.(combined[NUMBIDS+1:2*NUMBIDS,:]), tanh.(combined[end,:])
end
(mdl::RnnMlpMultiHeadModel)(x::Tuple) = mdl(x...)
#(mdl::RnnMlpMultiHeadModel)(x::Tuple,y) = nll(mdl(x),y)
function (mdl::RnnMlpMultiHeadModel)(x::Tuple, bid_gold, value_gold)
    #value_gold = atype(value_gold)
    bs = size(value_gold, 1)
    pol, actval, val = mdl(x)

    pol_loss = nll(pol, bid_gold)
    q_loss = mean(abs2, actval[bid_gold,:][1:size(actval,2)+1:end] .- value_gold)
    v_loss = mean(abs2, val .- value_gold)

    pol_loss + q_loss + v_loss
end


function RnnMlpMultiHeadModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64])
    RnnMlpMultiHeadModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, numLayers=lstmlayers),
        MLP(lstmhidden,mlphidden...,NUMBIDS*2+1)
    )
end

function DropoutRnnMlpMultiHeadModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            lstmhidden::Int=64, lstmlayers::Int=1, mlphidden::Vector{Int}=[64],
                            lstmdropout::Float64=.0, mlpdropout::Vector{Float64}=[.0,.0])
    RnnMlpMultiHeadModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        RNN(bidembed+cardembed+vulembed, lstmhidden, dropout=lstmdropout, numLayers=lstmlayers),
        Dropout(MLP(lstmhidden,mlphidden...,2*NUMBIDS+1), mlpdropout)
    )
end
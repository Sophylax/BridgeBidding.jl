struct SelfAttnMlpModel <: BridgePolicyModel
    cardembed::TransformerParts.Embed
    bidembed::TransformerParts.Embed
    vulembed::TransformerParts.Embed
    posencod::TransformerParts.PositionalEncoding
    totalencoder
    dense::Chain
end

getobserver(mdl::SelfAttnMlpModel) = MaskedHistoryObserver()

function (mdl::SelfAttnMlpModel)(hand, past, vul)
    h_embed = mdl.cardembed(hand)
    v_embed = mdl.vulembed(vul)
    p_embed = mdl.bidembed(past)
    p_embed = mdl.posencod(p_embed)
    c_embed = cat(p_embed, h_embed, v_embed, dims=2)
    #total_encoding = mdl.totalencoder(c_embed)
    total_encoding = mdl.totalencoder.layers[1].selfattn.layer(c_embed)
    total_encoding = mdl.totalencoder.layers[end](total_encoding)
    mdl.dense(total_encoding[:,1,:])
end
(mdl::SelfAttnMlpModel)(x::Tuple) = mdl(x...)
(mdl::SelfAttnMlpModel)(x::Tuple,y) = nll(mdl(x),y)
function SelfAttnMlpModel(;dmodel::Int=32, dff::Int=64, nheads::Int=2, nlayers::Int=1, edropout::Float64=.0,
                        maxlen::Int = 40, mlphidden::Vector{Int}=[64], mlpdropout::Vector{Float64}=[.0,.0])
    SelfAttnMlpModel(
        TransformerParts.Embed(52, dmodel),
        TransformerParts.Embed(NUMBIDS+1, dmodel),
        TransformerParts.Embed(4, dmodel),
        TransformerParts.PositionalEncoding(dmodel, maxlen),
        TransformerParts.Chain([TransformerParts.EncoderLayer(dmodel, dff, nheads, edropout) for n=1:nlayers]..., TransformerParts.LayerNorm(dmodel)),
        Dropout(MLP(dmodel,mlphidden...,NUMBIDS), mlpdropout)
    )
end
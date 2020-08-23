struct MlpModel <: BridgeModel
    cardembed::Linear
    bidembed::Embed
    vulembed::Embed
    dense::Chain
end

getobserver(mdl::MlpModel) = FixedHistoryObserver()

function (mdl::MlpModel)(hand, past, vul)
    h_embed = mdl.cardembed(hand)
    p_embed = mdl.bidembed(past)
    p_embed = reshape(p_embed, :, size(p_embed,3))
    v_embed = mdl.vulembed(vul)
    total_encoding = cat(h_embed,p_embed,v_embed,dims=1)
    mdl.dense(total_encoding)
end
(mdl::MlpModel)(x::Tuple) = mdl(x...)
(mdl::MlpModel)(x::Tuple,y) = nll(mdl(x),y)
function MlpModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            mlphidden::Vector{Int}=[64])
    MlpModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        MLP(cardembed+vulembed+(bidembed*_fixedhistorysize),mlphidden...,NUMBIDS)
    )
end

function DropoutMlpModel(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
                            mlphidden::Vector{Int}=[64],
                            mlpdropout::Vector{Float64}=[.0,.0])
    MlpModel(
        Linear(input=52, output=cardembed),
        Embed(vocab=NUMBIDS+1, embed=bidembed),
        Embed(vocab=4, embed=vulembed),
        Dropout(MLP(cardembed+vulembed+(bidembed*_fixedhistorysize),mlphidden...,NUMBIDS), mlpdropout)
    )
end
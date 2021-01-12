module TransformerParts
## Transformer magic
using Knet
import ..atype
using Statistics: mean, std

struct MaskedArray; array; mask; end
MaskedArray(array) = MaskedArray(array, trues(size(array)))

import Base: size, length, show, lastindex, getindex, reshape
size(m::MaskedArray,d...) = size(m.array,d...)
length(m::MaskedArray) = length(m.array)
show(io::IO,m::MaskedArray) = print(io,"MaskedArray($(typeof(m.array))($(join(size(m.array),','))),$(typeof(m.mask))($(m.mask===nothing ? "" : join(size(m.mask),','))))[$(m.array[1])⋯]")
lastindex(m::MaskedArray,d...)=lastindex(m.array,d...)
getindex(m::MaskedArray,d...)=(@assert size(m.array)==size(m.mask); MaskedArray(getindex(m.array,d...),getindex(m.mask,d...)))
reshape(m::MaskedArray,d...)=(@assert size(m.array)==size(m.mask); MaskedArray(reshape(m.array, d...),m.mask===nothing ? nothing : reshape(m.mask, d...)))

import Knet: nll, dropout
nll(g::MaskedArray, y::MaskedArray; o...)=nll(g.array, y.array .* y.mask; o...)
dropout(x::MaskedArray,p)=MaskedArray(dropout(x.array,p),x.mask)

import Base.Broadcast: broadcasted
broadcasted(+, x::MaskedArray, y::MaskedArray)=(@assert x.mask == y.mask; MaskedArray(x.array .+ y.array, x.mask))
broadcasted(+, x::MaskedArray, y)=MaskedArray(x.array .+ y, x.mask)
broadcasted(+, x, y::MaskedArray)=MaskedArray(x .+ y.array, y.mask)
broadcasted(relu, x::MaskedArray) = MaskedArray(relu.(x.array), x.mask)

import Base: cat
function cat(X::MaskedArray...; dims) where {T}
    arrys = map(X) do ma ma.array end
    masks = map(X) do ma ma.mask end

    MaskedArray(cat(arrys..., dims=dims), cat(masks..., dims=dims))
end

struct Transformer; srcvocab; tgtvocab; srcembed; tgtembed; encoder; decoder; generator; end

# TODO: Transformer options = atype, winit, binit, init?, separate attention_dropout?
function Transformer(srcvocab, tgtvocab; dmodel=512, dff=2048, nheads=8, nlayers=6, maxlen=5000, dropout=0) 
    posembed, droplayer = PositionalEncoding(dmodel, maxlen), Dropout(dropout)
    srcembed = Chain(Embed(length(srcvocab), dmodel), posembed, droplayer)
    tgtembed = Chain(Embed(length(tgtvocab), dmodel), posembed, droplayer)
    encoder = Chain([EncoderLayer(dmodel, dff, nheads, dropout) for n=1:nlayers]..., LayerNorm(dmodel))
    decoder = Chain([DecoderLayer(dmodel, dff, nheads, dropout) for n=1:nlayers]..., LayerNorm(dmodel))
    generator = Linear(dmodel, length(tgtvocab))
    Transformer(srcvocab, tgtvocab, srcembed, tgtembed, encoder, decoder, generator)
end

function (t::Transformer)(src, tgt; average=true) # (T1,B) = size(src); (T2,B2) = size(tgt); @assert B == B2; (X,V) = size(t.tgtembed.layers[1].w)
    enc = t.encoder(t.srcembed(src))              # @size enc (X,T1,B)
    tgt1,tgt2 = tgt[1:end-1,:], tgt[2:end,:]      # @size tgt1 (T2-1,B); @size tgt2 (T2-1,B)
    dec = t.decoder(t.tgtembed(tgt1), enc)        # @size dec (X,T2-1,B)
    gen = t.generator(dec)                        # @size gen (V,T2-1,B)
    (sumloss,numwords) = nll(gen, tgt2, average=false) # TODO: handle nll difference for different batchsizes.
    average ? sumloss/numwords : (sumloss,numwords)   # TODO: loss per word or per sequence?
end


## Chain

struct Chain; layers; end

function Chain(layer1, layer2, layers...)
    Chain((layer1, layer2, layers...))
end

function (l::Chain)(x, o...)
    for layer in l.layers
        x = layer(x, o...)
    end
    return x
end



## EncoderLayer

struct EncoderLayer; selfattn; feedforw; end

function EncoderLayer(dmodel::Int, dff::Int, nheads::Int, dropout)
    selfattn = MultiHeadAttention(dmodel, nheads, dropout)
    selfattn = SubLayer(selfattn, dmodel, dropout)
    feedforw = FeedForward(dmodel, dff, dropout)
    feedforw = SubLayer(feedforw, dmodel, dropout)
    EncoderLayer(selfattn, feedforw)
end

function (l::EncoderLayer)(x)
    l.feedforw(l.selfattn(x))
end


## DecoderLayer

struct DecoderLayer; selfattn; srcattn; feedforw; end

function DecoderLayer(dmodel::Int, dff::Int, nheads::Int, dropout)
    selfattn = MultiHeadAttention(dmodel, nheads, dropout, selfmask=true)
    selfattn = SubLayer(selfattn, dmodel, dropout)
    srcattn  = MultiHeadAttention(dmodel, nheads, dropout)
    srcattn  = SubLayer(srcattn,  dmodel, dropout)
    feedforw = FeedForward(dmodel, dff, dropout)
    feedforw = SubLayer(feedforw, dmodel, dropout)
    DecoderLayer(selfattn, srcattn, feedforw)
end

function (l::DecoderLayer)(y,x)
    l.feedforw(l.srcattn(l.selfattn(y), x))
end


## MultiHeadAttention

struct MultiHeadAttention; q; k; v; o; dropout; scale; selfmask; end

function MultiHeadAttention(dmodel::Int, nheads::Int, dropout; selfmask=false, scale=1/sqrt(dmodel÷nheads))
    @assert dmodel % nheads == 0
    dk = dmodel ÷ nheads
    q = Linear(dmodel,dk,nheads)
    k = Linear(dmodel,dk,nheads)
    v = Linear(dmodel,dk,nheads)
    o = Linear(dmodel,dmodel)
    MultiHeadAttention(q,k,v,o,dropout,scale,selfmask)
end

function (l::MultiHeadAttention)(q,k,v; keymask=nothing)    # inputs all batch-major
    # (Q1,K1,V1) = size.((q,k,v),(1,)); (Q2,K2,V2) = size.((l.q.w,l.k.w,l.v.w),(1,)); (H,T1,T2) = size.((l.q.w,k,q),(2,)); B = size(q,3); (O2,O1) = size(l.o.w)
    # @size     q (Q1,T2,B); @size     k (K1,T1,B);     @size v (V1,T1,B)
    # @size l.q.w (Q2,H,Q1); @size l.k.w (K2,H,K1); @size l.v.w (V2,H,V1); @assert K2 == Q2; @assert O1 == V2*H
    # query, keys and values:
    q,k,v = l.q(q),l.k(k),l.v(v)             # @size q (Q2,H,T2,B); @size k (K2,H,T1,B); @size v (V2,H,T1,B)
    q,v = permutedims.((q,v), ((1,3,2,4),))  # @size q (Q2,T2,H,B); @size v (V2,T1,H,B)
    k = permutedims(k, (3,1,2,4))            # @size k (T1,K2,H,B)
    # scores:
    s = bmm(k,q)                             # @size s (T1,T2,H,B)
    s = s * l.scale                          # @size s (T1,T2,H,B)
    s = attnmask(s, keymask, l.selfmask)     # @size s (T1,T2,H,B)
    s = softmax(s, dims=1)                   # @size s (T1,T2,H,B)
    s = dropout(s, l.dropout)                # This is where all implementations put attention_dropout.
    # context:
    c = bmm(v,s)                             # @size c (V2,T2,H,B)
    c = permutedims(c, (1,3,2,4))            # @size c (V2,H,T2,B)
    c = reshape(c, :, size(c,3), size(c,4))  # @size c (O1,T2,B)
    o = l.o(c)                               # @size o (O2,T2,B)
    return o
end

function (l::MultiHeadAttention)(q::MaskedArray,k::MaskedArray,v::MaskedArray)
    # We turn (Q,Tq,B),(K,Tk,B),(V,Tk,B) -> (V,Tq,B)
    # The query mask will be applied to the output mask, does not effect the attention calculation
    # Only the key/value mask will effect the inner score calculation
    # Target time masking requires a (T,T) mask, not compatible with input sizes, has to be generated inside
    @assert k.mask == v.mask
    @assert size(q.mask,1) == 1
    a = l(q.array, k.array, v.array, keymask = k.mask)
    MaskedArray(a, q.mask)
end

(l::MultiHeadAttention)(x)=l(x,x,x)
(l::MultiHeadAttention)(y,x)=l(y,x,x)

function attnmask(s, keymask, do_selfmask)   # s=(Tk,Tq,H,B) keymask=(1,Tk,B) selfmask=(T,T,1,1)
    mask = nothing
    if keymask !== nothing
        @assert size(keymask) == (1, size(s,1), size(s,4))
        mask = reshape(keymask, size(s,1), 1, 1, size(s,4))
    end
    if do_selfmask
        @assert size(s,1) == size(s,2)
        T = size(s,1)
        sm = [ key <= qry for key in 1:T, qry in 1:T ]  # qry should see up to its own position but no further
        if mask === nothing
            mask = reshape(sm, T, T, 1, 1)
        else
            mask = mask .& sm
        end
    end
    if mask === nothing
        return s
    else
        return s .+ oftype(s, -1e9 * .!mask)
    end
end


## FeedForward

function FeedForward(dmodel, dff, dropout)
    Chain(Linear(dmodel,dff), Relu(), Dropout(dropout), Linear(dff,dmodel))
end


## SubLayer

struct SubLayer; layer; norm; dropout; end

function SubLayer(layer, dmodel::Int, dropout::Number)
    SubLayer(layer, LayerNorm(dmodel), Dropout(dropout))
end

# The paper suggests l.norm(x+l.dropout(l.layer(x))), however x + l.dropout(l.layer(l.norm(x))) 
# is the default implementation in the code, see discussion on "LAYER NORMALIZATION" below.
function (l::SubLayer)(x, xs...)
    x .+ l.dropout(l.layer(l.norm(x), xs...))
end

## PositionalEncoding

struct PositionalEncoding; w; end

function PositionalEncoding(dmodel, maxlen; λ=10000)
    x = exp.((0:2:dmodel-1) .* -(log(λ)/dmodel)) * (0:maxlen-1)'
    pe = zeros(dmodel, maxlen)
    pe[1:2:end,:] = sin.(x)
    pe[2:2:end,:] = cos.(x)
    PositionalEncoding(atype(pe))
end

function (l::PositionalEncoding)(x)
    x .+ l.w[:,1:size(x,2)]
end


## Linear: generalizes mmul to more than 2 dims: (A...,B) x (B,C...) => (A...,C...)

struct Linear; w; b; end

function Linear(input::Int,outputs...; bias=true)
    Linear(param(outputs...,input),
           bias ? param0(outputs...) : nothing)
end

function (l::Linear)(x)
    W1,W2,X1,X2 = size(l.w)[1:end-1], size(l.w)[end], size(x,1), size(x)[2:end]; @assert W2===X1
    y = reshape(l.w,:,W2) * reshape(x,X1,:)
    y = reshape(y, W1..., X2...)
    if l.b !== nothing; y = y .+ l.b; end
    return y
end

function (l::Linear)(x::MaskedArray)
    (a,m) = (x.array, x.mask)
    @assert m===nothing || all(size(m,i) == 1 || size(m,i) == size(a,i) for i in 1:ndims(a))
    if m === nothing
        return MaskedArray(l(a), nothing)
    elseif size(m,1) == 1   # avoid mask multiplication if possible
        b = l(a)
        if ndims(b) > ndims(m)
            m = reshape(m, ntuple(i->1, ndims(b)-ndims(m))..., size(m)...)
        end
        return MaskedArray(b, m)
    else
        return MaskedArray(l(a .* oftype(a,m)), nothing)
    end
end

## Relu

struct Relu end

function (l::Relu)(x)
    relu.(x)
end

struct LayerNorm; a; b; ϵ; end

function LayerNorm(dmodel; eps=1e-6)
    a = param(dmodel; init=ones)
    b = param(dmodel; init=zeros)
    LayerNorm(a, b, eps)
end

function (l::LayerNorm)(x, o...)
    μ = mean(x,dims=1)
    σ = std(x,mean=μ,dims=1)
    l.a .* (x .- μ) ./ (σ .+ l.ϵ) .+ l.b # TODO: doing x .- μ twice?
end

function (l::LayerNorm)(x::MaskedArray, o...)
    MaskedArray(l(x.array), x.mask) # TODO: shouldn't normalization ignore masked values?
end


## Dropout

struct Dropout; p; end

function (l::Dropout)(x)
    dropout(x, l.p) # TODO: dropout normalization does not depend on masks?
end

struct Embed; w; end

function Embed(vocabsize,embedsize)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end

function (l::Embed)(x::MaskedArray)
    a = l(x.array)
    m = (x.mask === nothing ? nothing : reshape(x.mask, 1, size(x.mask)...))
    MaskedArray(a, m)
end


end
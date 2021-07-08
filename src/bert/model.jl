using IterTools, Base.Iterators, Printf, LinearAlgebra, Statistics, Knet
#using Knet: param, param0, atype, bmm, softmax, dropout, nll, relu
#using Statistics: mean, std
macro size(z, s); esc(:(@assert  size($z) == $s  string(summary($z),!=,$s))); end

## BERT

struct BERT; xvocab; yvocab; embed; layers; pred; end

function BERT(xvocab, yvocab; dmodel=512, dff=2048, nheads=8, nlayers=6, maxlen=5000, dropout=0) 
    posembed, droplayer = PositionalEncoding(dmodel, maxlen), Dropout(dropout)
    embed = Chain(Embed(length(xvocab), dmodel), posembed, droplayer)
    layers = Chain([BERTLayer(dmodel, dff, nheads, dropout) for n=1:nlayers]..., LayerNorm(dmodel))
    pred = Linear(dmodel, length(yvocab))
    BERT(xvocab, yvocab, embed, layers, pred)
end

function (b::BERT)(x)           ; (T,B) = size(x); (X,V) = size(b.embed.layers[1].w); Y = length(b.yvocab)
    y = b.embed(x)              ; @size y (X,T,B)
    y = b.layers(y)             ; @size y (X,T,B)
    y = y[:,1,:]                ; @size y (X,B)
    y = b.pred(y)               ; @size y (Y,B)
    return y
end

function (b::BERT)(x, c; average=true)  ; (T,B) = size(x); Y = length(b.yvocab); @size c (B,)
    y = b(x)                            ; @size y (Y,B)
    nll(y, c, average=average)
end

function loss(b::BERT, data; average=true)
    sum,cnt = 0,0
    for (x,y) in data
        (l,n) = b(x,y,average=false)
        sum += l; cnt += n
    end
    average ? (sum/cnt) : (sum,cnt)
end

loss(b::BERT, bd::BridgeData; o...) = loss(b, bd.data; o...)

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


## BERTLayer

struct BERTLayer; selfattn; feedforw; end

function BERTLayer(dmodel::Int, dff::Int, nheads::Int, dropout)
    selfattn = MultiHeadAttention(dmodel, nheads, dropout)
    selfattn = SubLayer(selfattn, dmodel, dropout)
    feedforw = FeedForward(dmodel, dff, dropout)
    feedforw = SubLayer(feedforw, dmodel, dropout)
    BERTLayer(selfattn, feedforw)
end

function (l::BERTLayer)(x)
    l.feedforw(l.selfattn(x))
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


## LayerNorm: https://arxiv.org/abs/1607.06450: Layer Normalization
# TODO: this is slow, need a kernel, maybe https://github.com/tensorflow/tensorflow/pull/6205/files

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


## Dropout

struct Dropout; p; end

function (l::Dropout)(x)
    dropout(x, l.p)
end


## Relu

struct Relu end

function (l::Relu)(x)
    relu.(x)
end


## Embed

struct Embed; w; end

function Embed(vocabsize,embedsize)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end


## PositionalEncoding

struct PositionalEncoding; w; end

function PositionalEncoding(dmodel, maxlen; λ=10000, atype=Knet.atype())
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


module SupervisedModel

using Knet

export BaselineClassifier

atype=gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

include("../types.jl")

struct Embed; w; end
Embed(;vocab::Int,embed::Int)=Embed(param(embed,vocab))
(e::Embed)(x) = e.w[:,x]

struct Linear; w; b; end
Linear(;input::Int, output::Int)=Linear(param(output,input), param0(output))
(l::Linear)(x) = l.w * mat(x,dims=1) .+ l.b

struct NonLinear; f; l; end
NonLinear(;input::Int, output::Int, activation::Function=relu)=NonLinear(activation, Linear(input=input, output=output))
(nl::NonLinear)(x) = nl.f.(nl.l(x))

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

function MLP(h...; activation=relu)
    hiddennonlinear(inp,outp) = NonLinear(input=inp, output=outp, activation=activation)
    layers = []
    append!(layers, hiddennonlinear.(h[1:end-2],h[2:end-1]))
    push!(layers, Linear(input=h[end-1],output=h[end]))
    Chain(layers...)
end

struct BaselineClassifier
	cardembed::Linear
	bidembed::Embed
	vulembed::Embed
	pastencoder::RNN
	dense::Chain
end

function (bc::BaselineClassifier)(hand, past, vul, batchsizes)
    h_embed = bc.cardembed(hand)
    p_embed = bc.bidembed(past)
    p_hidds = bc.pastencoder(p_embed, batchSizes=batchsizes)
    p_encod = p_hidds[:, last_indices(batchsizes)]
    v_embed = bc.vulembed(vul)
    total_encoding = cat(h_embed,p_encod,v_embed,dims=1)
    bc.dense(total_encoding)
end
(bc::BaselineClassifier)(x::Tuple) = bc(x...)
(bc::BaselineClassifier)(x,y) = nll(bc(x),y)

function last_indices(batchsizes)
	lastsize = 0
	rev_cursor = sum(batchsizes)
	indices = Int[]
	for bs in reverse(batchsizes)
		term = bs - lastsize
		append!(indices, rev_cursor-term+1:rev_cursor)
		rev_cursor -= bs
		lastsize = bs
	end
	indices
end

function BaselineClassifier(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
							lstmhidden::Int=64, mlphidden::Vector{Int}=[64])
    BaselineClassifier(
    	Linear(input=52, output=cardembed),
    	Embed(vocab=NUMBIDS+1, embed=bidembed),
    	Embed(vocab=4, embed=vulembed),
    	RNN(bidembed, lstmhidden),
    	MLP(cardembed+vulembed+lstmhidden,mlphidden...,NUMBIDS)
    )
end

end
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

struct Dropout; probability::Float64; end
(d::Dropout)(x) = dropout(x,d.probability)
function Dropout(chain::Chain, probabilities::Vector{Float64})
    layers = [l for l in chain.layers]
    for i in 1:length(layers)
        insert!(layers,2*i-1,Dropout(probabilities[i]))
    end
    Chain(layers...)
end

function last_indices_for_rnn(batchsizes)
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

struct Residual
    layer
end
(r::Residual)(x) = r.layer(x) .+ x
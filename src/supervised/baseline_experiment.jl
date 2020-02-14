module SupervisedBaselineExperiment

using Knet, ArgParse, Base.Iterators, IterTools

include("model.jl")
include("data.jl")

atype=gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
	s = ArgParseSettings()
    s.description = "Supervised Baseline Experiment for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--gamesfile"; arg_type=String; default="../../data/refinedgames.json"; help="location of the supervised games data")
        ("--batchsize"; arg_type=Int; default=64; help="minibatch size")
        ("--cardembed"; arg_type=Int; default=64; help="card embedding size")
        ("--bidembed"; arg_type=Int; default=64; help="bid embedding size")
        ("--vulembed"; arg_type=Int; default=64; help="vulnerability embedding size")
        ("--lstmhidden"; arg_type=Int; default=128; help="lstm hidden size")
        ("--mlphidden"; nargs='*'; arg_type=Int; default=[64]; help="mlp hidden layer sizes, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--epochs"; arg_type=Int; default=10; help="number of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--learningrate"; arg_type=Float64; default=0.001; help="Learning rate for Adam")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;gamesfile::String, epochs::Int=10, batchsize::Int=64,
			cardembed::Int=32, bidembed::Int=64, vulembed=64, lstmhidden::Int=64,
			mlphidden::Vector{Int}=[64], seed::Int=-1, learningrate::Float64=0.001)

	@info "Bridge Bidding Baseline Supervised Experiment"
	@info "Model Configuration" cardembed bidembed vulembed lstmhidden mlphidden
	@info "Other Hyperparameters" batchsize learningrate epochs

	if seed > 0
		@info "Setting RNG seed to $seed"
		Knet.seed!(seed)
	end

	dtrn, dtst = SupervisedData.generate_dataset(gamesfile, split_ratios=[9,1], batchsize=batchsize)

	@info "Data loaded" length(dtrn) length(dtst)

	@info "Creating new baseline model from scratch"
	model = SupervisedModel.BaselineClassifier(cardembed=cardembed, bidembed=bidembed, vulembed=vulembed,
								lstmhidden = lstmhidden, mlphidden=mlphidden)

	progress!(adam(model,ncycle(dtrn,epochs); lr=learningrate), steps=length(dtrn)) do p
        println(stderr)
        trn = accuracy(model,take(dtrn,div(length(dtrn),10)))
        tst = accuracy(model,dtst)
        #push!(logtrn,trn)
        #push!(logtst,tst)
        (trn=round(trn, digits=6), tst=round(tst, digits=6))
    end
end 

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end

end
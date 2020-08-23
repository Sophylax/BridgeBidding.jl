include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
    s = ArgParseSettings()
    s.description = "Supervised Experiment for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--gamesfile"; arg_type=String; default="../../data/refinedgames.json"; help="location of the supervised games data")
        ("--batchsize"; arg_type=Int; default=64; help="minibatch size")
        ("--epochs"; arg_type=Int; default=10; help="number of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--learningrate"; arg_type=Float64; default=0.001; help="Learning rate for Adam")
        ("--modelfile"; arg_type=String; help="file name to load and save the model"; required = true)
        ("--train"; action=:store_true; help="do training on the model")
        ("--eval"; action=:store_true; help="do evaluation on the model")
        ("--logfile"; arg_type=String; help="file name to save logs")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;gamesfile::String, epochs::Int=10, batchsize::Int=64,
            seed::Int=-1, learningrate::Float64=0.001, modelfile::String=nothing,
            train::Bool=false, eval::Bool=false, logfile::Union{Nothing, String}=nothing)

    if !train && !eval
        @error "Neither --train nor --eval is flagged. Exiting..."
        exit()
    end

    if !isnothing(logfile)
        default_logger = global_logger()
        file_logger = FileLogger(logfile)
        demux_logger = TeeLogger(default_logger, file_logger)
        global_logger(demux_logger)
    end

    @info "Bridge Bidding Supervised Experiment"
    if train
       @info "Training Hyperparameters" batchsize learningrate epochs
    end

    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
    end

    @info "Loading model from \"$modelfile\""
    model = Knet.load(modelfile, "model")

    dtrn, dtst = generate_supervised_bidset(gamesfile, split_ratios=[9,1], batchsize=batchsize)
    @info "Data loaded" length(dtrn) length(dtst)
    
    savefun = () -> Knet.save(modelfile, "model", model)

    evaluator = SupervisedBidEvaluator(take(dtrn,div(length(dtrn),10)), dtst, onnewbest = savefun)

    if train

        trainer = adam(model, dtrn; lr=learningrate) #SupervisedTrainer(dtrn,adam,learningrate=learningrate)

        if eval
            trainer = evaluate(trainer) do
                evaluator(model)
            end
        else

        end

        progress!(ncycle(trainer, epochs))

    elseif eval
        println(evaluator(model))
    end

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
    s = ArgParseSettings()
    s.description = "Actor-Critic Experiment for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--dummyfile"; arg_type=String; default="../../data/dd-500k.json"; help="location of the double dummy data")
        ("--epochs"; arg_type=Int; default=1; help="number of training epochs")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--learningrate"; arg_type=Float64; default=0.001; help="Learning rate for Adam")
        ("--actorfile"; arg_type=String; help="file name to load and save the model"; required = true)
        ("--criticfile"; arg_type=String; help="file name to load and save the model"; required = true)
        ("--fixedsides"; arg_type=String; default=""; help="sides for fixed actors, eg: ns")
        ("--train"; action=:store_true; help="do training on the model")
        ("--eval"; action=:store_true; help="do evaluation on the model")
        ("--logfile"; arg_type=String; help="file name to save logs")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;dummyfile::String, epochs::Int=1, batchsize::Int=64, seed::Int=-1, fixedsides::String="",
            learningrate::Float64=0.001, actorfile::String=nothing, criticfile::String=nothing,
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

    @info "Bridge Bidding Actor-Critic Experiment"
    if train
       @info "Training Hyperparameters" learningrate epochs
    end

    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
    end

    @info "Loading actor model from \"$actorfile\""
    actor = Knet.load(actorfile, "model")
    fixedactor = deepcopy(actor);

    @info "Loading critic model from \"$criticfile\""
    critic = Knet.load(criticfile, "model")

    for param in Knet.params(actor); param.opt = Adam(lr=learningrate); end;
    for param in Knet.params(critic); param.opt = Adam(lr=learningrate); end;

    ddtrn,ddtst = generate_doubledummy_gameset("BridgeBidding/data/dd-500k.txt", split_ratios=[499,1], batchsize=64);
    @info "Data loaded" length(ddtrn) length(ddtst)

    #Fixed side logic
    actors = Array{Any}(fill(nothing,4))
    if fixedsides != ""
        fixedsides = uppercase(fixedsides)
        @info "Fixing actors of given sides: $fixedsides"
        for ch in fixedsides
            idx = findfirst(isequal(ch), BridgeBidding.PLAYERCHAR)
            if idx == nothing
                @error "Unrecognized side to fix: $ch. Exiting..."
                exit()
            end
            actors[idx] = fixedactor
        end
    end
    
    @info "Warming up the game engine"
    playepisode(ddtst.games[1:batchsize], actormodels=fill(actor, 4))

    gse = GameScoreEvaluator(ddtst);
    ae = AdversarialEvaluator(ddtst);
    evalfun = ()->begin gse(actor); ae(actor, fixedactor); end

    if train

        trainer = ActorCritic(actor, critic, ddtrn, actors=actors)

        if eval
            trainer = timedevaluate(evalfun, trainer)
        else

        end

        progress!(ncycle(trainer, epochs))

    elseif eval
        evalfun()
    end

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
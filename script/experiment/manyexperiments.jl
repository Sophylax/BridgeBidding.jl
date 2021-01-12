include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
    s = ArgParseSettings()
    s.description = "Bridge Bidding stupid many experiment thing"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--fprefix"; arg_type=String; required = true)
        ("--code"; arg_type=Int; required = true)
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;fprefix::String, code::Int, seed::Int=-1)

    fixactor = Knet.load("experiments/2020-06-25-supervised-memorize-rnnmlp-mk4-64-1024/model.jld2", "model");

    name = fprefix * "-scratch"

    scratch = true
    actorloss = ActorLoss(FinalMinusValue)
    valloss = MonteCarloLoss
    epochs = 1
    opponent = SelfScheduler()
    passschedule = AnyScheduler([(1,true)])
    trainopt = ()->Rmsprop(lr=1e-3)
    entropy = 0
    game = DuplicateGame()

    #Opponent: (Self, Fixed, Dual, Staggered-Self, Model Zoo, Zoo+Self)
    if code == 1
        opponent = FixedScheduler(fixactor)
        name *= "-fixed"
    elseif code == 2
        opponent = DualScheduler(fixactor)
        name *= "-dual"
    elseif code == 3
        opponent = StaggeredScheduler(200)
        name *= "-staggered-200"
    elseif code == 4
        opponent = ZooScheduler(1000)
        name *= "-zoo-1000"
    elseif code == 5
        opponent = ZooScheduler(1000,self=true)
        name *= "-zooself-1000"
    else
        name *= "-self"
    end

    #Entropy (0, 1e-4,5e-4,1e-3,5e-3,1e-2)
    if code == 6
        entropy = 1e-4
        name *= "-entropy-1e-4"
    elseif code == 7
        entropy = 5e-4
        name *= "-entropy-5e-4"
    elseif code == 8
        entropy = 1e-3
        name *= "-entropy-1e-3"
    elseif code == 9
        entropy = 5e-3
        name *= "-entropy-5e-3"
    elseif code == 10
        entropy = 1e-2
        name *= "-entropy-1e-2"
    else
        name *= "-entropy-0"
    end

    #Optimizer & LR (Rmsprop; 1e-3, 1e-2, 5e-3, 5e-4, 1e-4)
    if code == 11
        trainopt = ()->Rmsprop(lr=1e-2)
        name *= "-lr-1e-2"
    elseif code == 12
        trainopt = ()->Rmsprop(lr=5e-3)
        name *= "-lr-5e-3"
    elseif code == 13
        trainopt = ()->Rmsprop(lr=5e-4)
        name *= "-lr-5e-4"
    elseif code == 14
        trainopt = ()->Rmsprop(lr=1e-4)
        name *= "-lr-1e-4"
    else
        name *= "-lr-1e-3"
    end


    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
        name *= "-seed-$seed"
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
        name *= "-seed-$seed"
    end

    @info "Folder name: $name"
    mkpath(name)

    default_logger = global_logger()
    file_logger = FileLogger("$name/run.log")
    demux_logger = TeeLogger(default_logger, file_logger)
    global_logger(demux_logger)

    ddtrn = generate_doubledummy_gameset("BridgeBidding/data/dd-train-5120k.txt", batchsize=64)[1];
    ddtst = generate_doubledummy_gameset("BridgeBidding/data/dd-test-10k.txt", batchsize=64)[1];

    gse = GameScoreEvaluator(ddtst);
    ae = AdversarialEvaluator(ddtst);
    nce = NonCompEvaluator(ddtst);

    model = if scratch
        RnnMlpMultiHeadModel(cardembed=64,bidembed=64,vulembed=64,lstmhidden=1024,mlphidden=[1024]);
    else
        actor = Knet.load("experiments/2020-06-25-supervised-memorize-rnnmlp-mk4-64-1024/model.jld2", "model");
        critic = Knet.load("experiments/2020-08-13-supervised-memorize-value-64-1024/model.jld2", "model");
        BridgeMergedModel(policy = actor, value = critic)
    end

    for param in Knet.params(model); param.opt = trainopt(); end;

    evalfun = ()->begin println(stderr); gse(model); ae(model, fixactor); end

    trainer = ActorCritic(model, ddtrn, entropy=entropy, actorloss=actorloss, criticloss = valloss, epochs = epochs, passscheduler = passschedule, opponent = opponent, gameformat = game);
    trainer = evaluate(evalfun, trainer, cycle=200);
    progress!(trainer);
end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
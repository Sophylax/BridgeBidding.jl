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
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;fprefix::String, seed::Int=-1)

    fixactor = Knet.load("experiments/2020-06-25-supervised-memorize-rnnmlp-mk4-64-1024/model.jld2", "model");

    name = fprefix


    @info "Folder name: $name"

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
    mkpath(name)

    default_logger = global_logger()
    file_logger = FileLogger("$name/run.log")
    demux_logger = TeeLogger(default_logger, file_logger)
    global_logger(demux_logger)

    ddtrn = generate_doubledummy_gameset("BridgeBidding/data/dd-train-5120k.txt", batchsize=64)[1];
    ddtst = generate_doubledummy_gameset("BridgeBidding/data/dd-test-10k.txt", batchsize=64)[1];

    save_iteration = 0;

    genericsave = ()->begin
        Knet.save("$name/model$save_iteration.jld2", "model", model)
        save_iteration += 1;
    end

    gse_save = ()->begin
        Knet.save("$name/gse_best.jld2", "model", model)
    end

    ae_save = ()->begin
        Knet.save("$name/ae_best.jld2", "model", model)
    end

    gse = GameScoreEvaluator(ddtst, onnewbest=gse_save);
    ae = AdversarialEvaluator(ddtst, onnewbest=ae_save);

    model = RnnMlpMultiHeadModel(cardembed=64,bidembed=64,vulembed=64,lstmhidden=1024,mlphidden=[1024]);
    #actor = Knet.load("experiments/2020-06-25-supervised-memorize-rnnmlp-mk4-64-1024/model.jld2", "model");
    #critic = Knet.load("experiments/2020-08-13-supervised-memorize-value-64-1024/model.jld2", "model");
    #model = BridgeMergedModel(policy = actor, value = critic)

    for param in Knet.params(model); param.opt = Rmsprop(lr=7e-4); end;

    evalfun = ()->begin println(stderr); gse(model); ae(model, fixactor); end

    trainer = ActorCritic(model, ncycle(ddtrn, 30), entropy=5e-4, actorloss=ProximalLoss(FinalMinusValue), criticloss = MonteCarloLoss, epochs = 3, passscheduler = AnyScheduler([(1,true)]));
    trainer = evaluate(evalfun, trainer, cycle=200);
    trainer = evaluate(genericsave, trainer, cycle=4000);
    progress!(trainer);
end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
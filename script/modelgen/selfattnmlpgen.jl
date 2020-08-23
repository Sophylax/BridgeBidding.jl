include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Logging, LoggingExtras

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
	s = ArgParseSettings()
    s.description = "RNN Self Attention Model Generator for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--dmodel"; arg_type=Int; default=32; help="encoder model hidden size")
        ("--dff"; arg_type=Int; default=64; help="encoder model feedforward hidden size")
        ("--nheads"; arg_type=Int; default=2; help="encoder model attention head count")
        ("--nlayers"; arg_type=Int; default=1; help="encoder model layer count")
        ("--maxlen"; arg_type=Int; default=40; help="positional encoding max length")
        ("--mlphidden"; nargs='*'; arg_type=Int; default=[64]; help="mlp hidden layer sizes, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--mlpdropout"; nargs='*'; arg_type=Float64; default=[.0,.0]; help="mlp dropout rates, must be one more than hidden layer numbers")
        ("--edropout"; arg_type=Float64; default=.0; help="encoder dropout rate")
        ("--logfile"; arg_type=String; help="file name to save logs")
        ("--savefile"; arg_type=String; help="file name to save the model"; required = true)
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;dmodel::Int=32, dff::Int=64, nheads::Int=2, nlayers::Int=1, edropout::Float64=.0,
            maxlen::Int = 40, mlphidden::Vector{Int}=[64], mlpdropout::Vector{Float64}=[.0,.0],
            savefile::String=nothing, logfile::Union{Nothing, String}=nothing, seed::Int=-1)

    if !isnothing(logfile)
        default_logger = global_logger()
        file_logger = FileLogger(logfile)
        demux_logger = TeeLogger(default_logger, file_logger)
        global_logger(demux_logger)
    end

	@info "RNN Self Attention Model Generator for Bridge Bidding"

    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
    end

    @info "Model Configuration" dmodel dff nheads nlayers maxlen mlphidden=string(mlphidden)
    if edropout + sum(mlpdropout) > 0
        @info "Dropout Configuration" edropout mlpdropout=string(mlpdropout)
    end

    model = SelfAttnMlpModel(dmodel=dmodel, dff=dff, nheads=nheads, nlayers=nlayers,
                            edropout=edropout, maxlen=maxlen, mlphidden=mlphidden,
                            mlpdropout=mlpdropout)

    @info "Saving model to \"$savefile\""
    Knet.save(savefile, "model", model)

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
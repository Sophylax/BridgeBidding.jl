include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Logging, LoggingExtras

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})
	s = ArgParseSettings()
    s.description = "MLP Model Generator for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--cardembed"; arg_type=Int; default=64; help="card embedding size")
        ("--bidembed"; arg_type=Int; default=64; help="bid embedding size")
        ("--vulembed"; arg_type=Int; default=64; help="vulnerability embedding size")
        ("--mlphidden"; nargs='*'; arg_type=Int; default=[64]; help="mlp hidden layer sizes, e.g. --hidden 128 64 for a net with two hidden layers")
        ("--mlpdropout"; nargs='*'; arg_type=Float64; default=[.0,.0]; help="mlp dropout rates, must be one more than hidden layer numbers")
        ("--logfile"; arg_type=String; help="file name to save logs")
        ("--savefile"; arg_type=String; help="file name to save the model"; required = true)
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;cardembed::Int=32, bidembed::Int=64, vulembed=64,
            mlphidden::Vector{Int}=[64], mlpdropout::Vector{Float64}=[.0,.0],
            savefile::String=nothing, logfile::Union{Nothing, String}=nothing, seed::Int=-1)

    if !isnothing(logfile)
        default_logger = global_logger()
        file_logger = FileLogger(logfile)
        demux_logger = TeeLogger(default_logger, file_logger)
        global_logger(demux_logger)
    end

	@info "MLP Model Generator for Bridge Bidding"

    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
    end

    @info "Model Configuration" cardembed bidembed vulembed mlphidden=string(mlphidden)
    if sum(mlpdropout) > 0
        @info "Dropout Configuration" mlpdropout=string(mlpdropout)
    end

    if sum(mlpdropout) > 0
        model = DropoutMlpModel(cardembed=cardembed, bidembed=bidembed, vulembed=vulembed,
                            mlphidden=mlphidden, mlpdropout=mlpdropout)
    else
        model = MlpModel(cardembed=cardembed, bidembed=bidembed, vulembed=vulembed,
                            mlphidden=mlphidden)
    end


    @info "Saving model to \"$savefile\""
    Knet.save(savefile, "model", model)

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
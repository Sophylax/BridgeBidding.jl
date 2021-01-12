import Pkg
required = ["ArgParse", "Glob", "IterTools", "JSON", "Knet", "LoggingExtras", "StatsBase"]
#isinstalled(pkg::String) = any(x -> x.name == pkg && x.is_direct_dep, values(Pkg.dependencies())) NOT WORKING in 1.3
installed = keys(Pkg.installed())
for pkg in required
    if !(pkg in installed)
        Pkg.add(pkg)
    end
end

include("src/BridgeBidding.jl")
using .BridgeBidding
import .BridgeBidding: PLAYERCHAR, bid!, prettyprinthands, prettyprintauction, prettyprintdealer, legalbids, prettyprintvulnerable, SUITCHAR, makebid, winningbid, declarer

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras, Glob, Random, StatsBase

main(args::AbstractString) = main(split(args))

function main(args::Vector{<:AbstractString})
    s = ArgParseSettings()
    s.description = "Manual Playing Interface for Bridge Bidding"
    s.exc_handler=ArgParse.debug_handler

    @add_arg_table s begin
        ("--gamefile"; arg_type=String; default="tournament.pbn"; help="location of PBN file, undefined plays a single random game")
        ("--seed"; arg_type=Int; default=-1; help="random number seed: use a nonnegative int for repeatable results")
        ("--actor"; arg_type=String; default="models\\ppo3entropy1e-3actor.jld2"; help="file name to load the model")
        ("--manualsides"; arg_type=String; default=""; help="sides for human actor(s), eg: ns")
        ("--logfile"; arg_type=String; default="tournament.log"; help="file name to save logs")
        ("--firstgame"; arg_type=Int; default=-1; help="the game number to start the game play, negative number means the actual first game in the file")
        ("--lastgame"; arg_type=Int; default=-1; help="the game number to end the game play, negative number means the actual last game in the file")
        ("--probabilistic"; action=:store_true; help="use the model policy probabilisticly instead of choosing the most probable option")
    end

    main(; parse_args(args, s, as_symbols=true)...)
end

function main(;gamefile::String="tournament.pbn", seed::Int=-1, manualsides::String="", firstgame::Int=-1, lastgame::Int=-1,
            actor::String="models\\ppo3entropy1e-3actor.jld2", logfile::Union{Nothing, String}="tournament.log", probabilistic::Bool=false)

    if firstgame >= 0 && lastgame >= 0 && firstgame > lastgame
        @error "First game -$firstgame- is later than the last game -$lastgame."
        exit()
    end

    if !isnothing(logfile)
        default_logger = global_logger()
        file_logger = FileLogger(logfile, append=true)
        demux_logger = TeeLogger(default_logger, file_logger)
        global_logger(demux_logger)
    end

    @info "Manual Playing Interface for Bridge Bidding"

    if seed > 0
        @info "Setting RNG seed to $seed"
        Knet.seed!(seed)
    else
        seed = floor(Int,time())
        @info "Setting RNG seed to current unix time: $seed"
        Knet.seed!(seed)
    end

    @info "Loading actor model from \"$actor\""
    actor = Knet.load(actor, "model")

    if probabilistic
        @info "Policy usage mode: Probabilistic"
    else
        @info "Policy usage mode: Greedy"
    end

    #Fixed side logic
    manual_sides = []
    if manualsides != ""
        manualsides = uppercase(manualsides)
        @info "Manual actors are for given sides: $manualsides"
        for ch in manualsides
            idx = findfirst(isequal(ch), PLAYERCHAR)
            if idx == nothing
                @error "Unrecognized side to fix: $ch. Exiting..."
                exit()
            end
            push!(manual_sides, idx)
        end
    else
        @info "No manual sides are given: Self-play mode."
    end

    automode = length(manualsides) == 0

    games = []
    if gamefile == ""
        for lal in 1:4096
        deck = shuffle(1:52)
        dealer = rand(1:4)
        state = BridgeState([deck[1:13], deck[14:26], deck[27:39], deck[40:52]], starting_player=dealer)
        push!(games, state)
        end
    else
        if lowercase(gamefile)[end-3:end] == ".pbn"
            @info "Loading Matches from $gamefile"
            states = BridgeStatesFromPBN(open(f->read(f, String), gamefile))
            append!(games, states)
        else
            error("Invalid file name: $gamefile")
        end
    end

    if firstgame < 0
        firstgame = 1
    end
    if lastgame < 0
        lastgame = length(games)
    end

    @info "Warming up the game engine"
    actor(BridgeBidding.Observation(games[1]))

    for (i, game) in collect(enumerate(games))[firstgame:lastgame]

        println("\nGame $i/$(length(games)):\n")

        prettyprinthands(game)
        if !automode; prettyprintdealer(game); end
        prettyprintvulnerable(game)
        println()

        while !game.terminated
            if game.player in manual_sides
                while true
                    print("$(PLAYERCHAR[game.player]) bids: ")
                    input = uppercase(readline(stdin))
                    action = 0
                    if input == "X"
                        action = 37
                    elseif input == "XX"
                        action = 38
                    elseif length(input) > 0 && input[1] == 'P' && length(input) <= 4
                        action = 36
                    elseif length(input) == 2
                        level = tryparse(Int, string(input[1]))
                        trump = findfirst(x->x==input[2], SUITCHAR)
                        if level == nothing || trump == nothing
                            @warn "Invalid input: $input"
                            continue
                        elseif level < 1 || level > 7
                            @warn "Invalid bid level: $level"
                            continue
                        else
                            action = makebid(level, trump)
                        end
                    else
                        @warn "Invalid input: $input"
                        continue
                    end

                    if (action in legalbids(game))
                        bid!(game, action)
                        break
                    else
                        @warn "Illegal move"
                        continue
                    end
                end
            else
                obs = BridgeBidding.Observation(game)
                policy = Array(softmax(reshape(actor(obs), :)))
                legal = legalbids(game)
                action = probabilistic ? sample(legal, Weights(policy[legal])) : legal[findmax(policy[legal])[2]]
                if !automode
                    print("$(PLAYERCHAR[game.player]) bids: ")
                    if action == 36
                        print("Pass")
                    elseif action == 37
                        print("Double")
                    elseif action == 38
                        print("Redouble")
                    else
                        print(BridgeBidding.bidlevelchar(action))
                        print(BridgeBidding.bidtrumpchar(action))
                    end
                    #print(" - Probability: $(100*policy[action]/sum(policy[legal]))%")
                    println(".")
                end
                bid!(game, action)
            end
        end

        if automode; prettyprintauction(game); end
        contract = all(isequal(36), game.history) ? 0 : winningbid(game)
        decl = contract == 0 ? 0 : declarer(game, contract)
        @debug "Auction completed. Details: " game_number=i game.hands game.starting_player game.nsvul game.ewvul game.history contract, decl

    end

    if !isnothing(logfile)
        global_logger(global_logger().loggers[1])
    end

    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
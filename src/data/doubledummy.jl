struct DoubleDummy
	hands #(Array of four arrays, each having 13 integers for WEST NORTH EAST SOUTH respectively
	tricks #2D array of number of tricks taken by NS for all trumps and leads, eg. tricks[HEARTS,NORTH]
end

mutable struct DoubleDummySet
    games::Array{DoubleDummy,1}
    batchsize::Int
    ninstances::Int
    shuffled::Bool
end

function DoubleDummySet(games; batchsize::Int=32, shuffled::Bool=true)
    ninstances = length(games)
    DoubleDummySet(games, batchsize, ninstances, shuffled)
end

function BridgeState(dd::DoubleDummy; starting_player = WEST, nsvul = false, ewvul = false)
    BridgeState(dd.hands, [], starting_player, starting_player, nsvul, ewvul, false, dd.tricks)
end

function Base.iterate(d::DoubleDummySet, state=ifelse(d.shuffled, randperm(d.ninstances), 1:d.ninstances))
    n = length(state)
    n == 0 && return nothing
    
    batchsize = min(d.batchsize, n)
    idx, new_state = state[1:batchsize], state[batchsize+1:end]
    games = d.games[idx]

    return (games, new_state)
end

Base.length(d::DoubleDummySet) = Int(ceil(d.ninstances/d.batchsize))

function generate_doubledummy_gameset(games_file::String; split_ratios::Vector{Int}=[1],
                                    batchsize::Int=32, shuffled::Bool=true,
                                    mix_games::Bool=true, mix_seed::Int=5318008)
	@info "Loading double dummy game data from \"$games_file\""
	games = open(games_file, "r") do io
        readdoubledummy(io)
    end
	games_length = length(games)

	if mix_games
        if mix_seed > 0
            rng = MersenneTwister(mix_seed)
            shuffle!(rng, games)
        else
		  shuffle!(games)
        end
	end

	split_denom = sum(split_ratios)
	split_pos = map(cumsum(split_ratios)) do relative_pos
		floor(Int, relative_pos * games_length / split_denom)
	end
	pushfirst!(split_pos, 0)
	split_ranges = map(1:length(split_ratios)) do i
		(split_pos[i], split_pos[i+1])
	end
	split_data = map(split_ranges) do (range_start, range_end)
		raw_games = games[range_start+1:range_end]
		DoubleDummySet(raw_games, batchsize=batchsize, shuffled=shuffled)
	end
end

function readdoubledummy(stream::IO)
    data = []
    for line in eachline(stream)
        @assert length(line) == 88
        (hands,results) = split(line,':')
        hands = split(hands, ' ')
        h = [UInt8[] for i=1:4]
        for player in 1:4 # w,n,e,s
            suits = reverse(split(hands[player],'.'))
            for suit in 1:4 # c,d,h,s
                for char in collect(suits[suit])
                    value = findfirst(isequal(char), CARDCHAR)
                    @assert value > 0
                    card = makecard(suit,value)
                    push!(h[player], card)
                end
            end
        end
        r = zeros(UInt8,4,5)
        for trump in 1:5 # c,d,h,s,n
            for leader in 1:4 # w,n,e,s
                r[leader,trump] = hex2int(results[5-leader+4*(5-trump)])
            end
        end
        push!(data, DoubleDummy(h,r))
    end
    return data
end
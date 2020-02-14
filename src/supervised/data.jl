module SupervisedData

using Knet, JSON, Random

export SupervisedBid, SupervisedGamesData, generate_dataset

atype=gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

include("../types.jl")

struct SupervisedBid
    hand::Array{Int}
    past::Array{Int}
    vul::Tuple{Int, Int}
    bid::Int
end

function processbid(bid::String)
    if isdigit(bid[1]) #normal bid
        makebid(
            parse(Int, bid[1]),
            findfirst(isequal(bid[2]), SUITCHAR))
    elseif bid == "Pass"
        PASS
    elseif bid == "X"
        DOUBLE
    elseif bid == "XX"
        REDOUBLE
    elseif bid == "^S"
        PASS
    else
        error("Invalid bid: $bid")
    end
end

processauction(auction::Array) = map(processbid, filter(x->x!="*",auction))

processcard(card::String) = makecard(
    findfirst(isequal(card[2]), SUITCHAR),
    findfirst(isequal(card[1]), CARDCHAR))

processhand(hand::Array) = map(processcard, hand)

function processvul(dealer::Int, vul::String)
    if vul == "None"
        (0, 0)
    elseif vul == "Both"
        (1, 1)
    elseif occursin(PLAYERCHAR[dealer], vul)
        (1, 0)
    else
        (0, 1)
    end
end
        

function processgame(game)
    auction = processauction(game["auction"])
    
    if NUMBIDS in auction return SupervisedBid[]; end
    
    player = findfirst(isequal(game["dealer"][1]), PLAYERCHAR)
    hands = map(x->processhand(game["deal_verbose"][string(x)]), PLAYERCHAR)
    vul = processvul(player, game["vul"])
    
    supervised_data = SupervisedBid[]
    
    for (i, bid) in enumerate(auction)
        push!(supervised_data, SupervisedBid(hands[player], auction[1:i-1], vul, bid))
        
        player = mod1(player+1,4)
        vul = (vul[2], vul[1])
    end
    
    supervised_data
end

mutable struct SupervisedGamesData
    bids::Array{SupervisedBid,1}
    batchsize::Int
    ninstances::Int
    shuffled::Bool
end

function SupervisedGamesData(games; batchsize::Int=32, shuffled::Bool=true)
    data = collect(Iterators.flatten(map(processgame, games)));
    ninstances = length(data)
    return SupervisedGamesData(data, batchsize, ninstances, shuffled)
end

function Base.iterate(d::SupervisedGamesData, state=ifelse(d.shuffled, randperm(d.ninstances), 1:d.ninstances))
    n = length(state)
    n == 0 && return nothing
    
    batchsize = min(d.batchsize, n)
    idx, new_state = state[1:batchsize], state[batchsize+1:end]
    bids = d.bids[idx]
    sort!(bids, by=x->length(x.past), rev=true)
    
    longest = length(bids[1].past)
    hands = atype(zeros(Int, 52, batchsize))
    pasts = ones(Int, batchsize) .* 39 #Int[]
    batchsizes = zeros(Int, longest+1)
    batchsizes[1] = batchsize
    vuls = zeros(Int, batchsize)
    golds = zeros(Int, batchsize)

    for i in 1:batchsize
        for c in bids[i].hand
            hands[c,i] = 1
        end
        vuls[i] = (bids[i].vul[1]*2)+bids[i].vul[2]+1
        golds[i] = bids[i].bid
    end

    for t in 1:longest
    	for i in 1:batchsize
    		length(bids[i].past) < t && break
    		push!(pasts, bids[i].past[t])
    		batchsizes[t+1] +=1
    	end
    end

    return (((hands, pasts, vuls, batchsizes), golds), new_state)
end

Base.length(d::SupervisedGamesData) = Int(ceil(d.ninstances/d.batchsize))

function generate_dataset(games_file::String; split_ratios::Vector{Int}=[1],
							batchsize::Int=32, shuffled::Bool=true, mix_games::Bool=true)
	@info "Loading supervised game data from \"$games_file\""
	games = JSON.parsefile(games_file)
	games_length = length(games)

	if mix_games
		shuffle!(games)
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
		SupervisedGamesData(raw_games, batchsize=batchsize, shuffled=shuffled)
	end
end

end
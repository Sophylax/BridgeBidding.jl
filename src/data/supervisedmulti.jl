struct SupervisedMulti
    obs::Observation
    bid::Int
    value::Number
end

function SupervisedMulti(hand::Array{Int}, past::Array{Int}, vul::Tuple{Int, Int}, bid::Int, value::Number)
    SupervisedMulti(Observation(hand, past, vul), bid, value)
end

function Base.getproperty(sb::SupervisedMulti, v::Symbol)
    if v == :hand
        return sb.obs.hand
    elseif v == :past
        return sb.obs.past
    elseif v == :vul
        return sb.obs.vul
    else
        return getfield(sb, v)
    end
end

#return NS
function processscore(scorestr, declarer)
    parts = split(scorestr)
    if length(parts) == 1
        score = parse(Int, parts[1])
        if iseven(declarer)
            return score
        else
            return -score
        end
    elseif length(parts) == 2
        score = parse(Int, parts[2])
        if parts[1] == "NS"
            return score
        elseif parts[1] == "EW"
            return -score
        else
            error("Unexpected scorestr: $scorestr")
        end
    elseif length(parts) == 4
        score = parse(Int, parts[2])
        if parts[1] == "NS"
            return score
        elseif parts[1] == "EW"
            return -score
        else
            error("Unexpected scorestr: $scorestr")
        end
    else
        error("Unexpected scorestr: $scorestr")
    end
end


function processgameformulti(game)
    auction = processauction(game["auction"])
    
    if NUMBIDS in auction; return SupervisedMulti[]; end
    if game["score"] == ""; return SupervisedMulti[]; end
    
    player = findfirst(isequal(game["dealer"][1]), PLAYERCHAR)
    hands = map(x->processhand(game["deal_verbose"][string(x)]), PLAYERCHAR)
    vul = processvul(player, game["vul"])
    score = 0
    if game["score"] != "0" && game["score"] != "NS 0" && game["score"] != "EW 0" &&
            game["score"] != "NS 0 EW 0" && game["score"] != "EW 0 NS 0"
        if game["declarer"][1] == '^'; game["declarer"] = game["declarer"][2:end]; end 
        declarer = findfirst(isequal(game["declarer"][1]), PLAYERCHAR)
        score = processscore(game["score"], declarer)
    end
    parscore = game["dummypar"]["score"]
    paradvantage = imps(score-parscore)/24

    if isodd(player)
        paradvantage = -paradvantage
    end

    supervised_data = SupervisedMulti[]
    
    for (i, bid) in enumerate(auction)
        push!(supervised_data, SupervisedMulti(hands[player], auction[1:i-1], vul, bid, paradvantage))
        
        player = mod1(player+1,4)
        vul = (vul[2], vul[1])
        paradvantage = -paradvantage
    end
    
    supervised_data
end

mutable struct SupervisedMultiSet
    multis::Array{SupervisedMulti,1}
    batchsize::Int
    ninstances::Int
    shuffled::Bool
end

function SupervisedMultiSet(games; batchsize::Int=32, shuffled::Bool=true)
    data = collect(Iterators.flatten(map(processgameformulti, games)));
    ninstances = length(data)
    return SupervisedMultiSet(data, batchsize, ninstances, shuffled)
end

function Base.iterate(d::SupervisedMultiSet, state=ifelse(d.shuffled, randperm(d.ninstances), 1:d.ninstances))
    n = length(state)
    n == 0 && return nothing
    
    batchsize = min(d.batchsize, n)
    idx, new_state = state[1:batchsize], state[batchsize+1:end]
    multis = d.multis[idx]

    sort!(multis, by=x->length(x.past), rev=true)

    return (([bid.obs for bid in multis],[bid.bid for bid in multis],atype([bid.value for bid in multis])),new_state)
end

Base.length(d::SupervisedMultiSet) = Int(ceil(d.ninstances/d.batchsize))

function generate_supervised_multiset(games_file::String; split_ratios::Vector{Int}=[1],
							batchsize::Int=32, shuffled::Bool=true,
                            mix_games::Bool=true, mix_seed::Int=5318008)
	@info "Loading supervised game data from \"$games_file\""
	games = JSON.parsefile(games_file)
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
		SupervisedMultiSet(raw_games, batchsize=batchsize, shuffled=shuffled)
	end
end
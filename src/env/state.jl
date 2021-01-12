"""
    BridgeState(hands, history, player, starting_player, nsvul, ewvul, terminated, tricks, parscore, nonzeropass)
    BridgeState(hands, tricks = nothing; starting_player = WEST, nsvul = false, ewvul = false, nonzeropass = false)

Structure for representing a bridge bidding state.

# Fields
- `hands::Vector{Vector{Int}}`: Vector of four Vectors, one for each player/side. Each inner Vector contains 13 integers for the cards.
- `history::Vector{Int}`: Vector of bids made until this point.
- `player::Int`: Currently acting player.
- `starting_player::Int`: The first player to bid.
- `nsvul::Bool`: Vulnerability information for N/S.
- `ewvul::Bool`: Vulnerability information for E/W.
- `tricks::Union{Array{Int, 2},Nothing}`: Number of tricks taken by North-South for all trumps and leads. First dimension is for trumps and second one is for leads.
May also be nothing, indicating a game without double dummy data present.
- `parscore::Union{Int, Nothing}`: Stored par score, if it's calculated before.
- `nonzeropass::Bool`: Whether the state gives negative par to all pass game.
"""
mutable struct BridgeState
	hands #(Array of four arrays, each having 13 integers for WEST NORTH EAST SOUTH respectively
	history #Array of past bids made until this point
	player #Integer designating the next player
	starting_player #Integer designating the starting player
	nsvul #Boolean for NS vulnerability
	ewvul #Boolean for EW vulnerability
	terminated #Boolean for quick access if the episode ended
	tricks #2D array of number of tricks taken by NS for all trumps and leads, eg. tricks[NORTH, TRUMP]
			#May be nothing if double dummy data is not provided
	parscore
	nonzeropass
end

function BridgeState(hands, tricks = nothing; starting_player = WEST, nsvul = false, ewvul = false, nonzeropass = false)
	BridgeState(hands, [], starting_player, starting_player, nsvul, ewvul, false, tricks, nothing, nonzeropass)
end

"""
    BridgeState(; args...)

Randomly generate a deal and construct a BridgeState for that deal. No double dummy data.
"""
function BridgeState(; args...)
	deck = shuffle(1:52)
	hands = [deck[1:13], deck[14:26], deck[27:39], deck[40:52]]
	BridgeState(hands; args...)
end

"""
    nextplayer!(state::BridgeState)

Change the current player of the bridge state to the next one.
"""
function nextplayer!(state::BridgeState)
	state.player = mod1(state.player + 1, 4)
end

"""
    regularbids(state::BridgeState)

Filter and return the list of contract bids in the history.
"""
function regularbids(state::BridgeState)
    filter(x -> x <= TRUMPBIDS, state.history)
end

"""
    winningbid(state::BridgeState)

Return the largest contract bid made, or return zero if no contract bid is made.
"""
function winningbid(state::BridgeState)
	regulars = regularbids(state)
	if length(regulars) > 0
		return maximum(regulars)
	else
		return 0
	end
end

#Declarer of a bid. Get the bid's maker, find who in that partnership declared that trump
"""
    declarer(state::BridgeState, bid)

Given a bid, find and return the declarer of that bid.

A declarer is the player in the partnership that made the bid in question who made the first contract bid with the bid's trump.
"""
function declarer(state::BridgeState, bid)
	@assert bid in state.history
	maker = findfirst(x -> x==bid, state.history)

	declarer = findfirst(collect(enumerate(state.history))) do (i,x)
		x < PASS && i%2 == maker%2 && bidtrump(x)==bidtrump(bid)
	end
	no_of_player_changes = declarer - 1
	effective_player_changes = no_of_player_changes % 4
	mod1(state.starting_player + effective_player_changes, 4)
end

"""
    legalbids(state::BridgeState)

Return a list of legally allowed bids for the current player.

The list consists of the following:
- All contract bids larger than the largest contract bid.
- If the last non-pass bid was a contract bid made by the opponents, Double is included.
- If the last non-pass bid was a double made by the opponents, Redouble is included.
- Pass
"""
function legalbids(state::BridgeState)
	best = winningbid(state)
	bids = collect((best + 1):TRUMPBIDS)

	#TODO: Add agreeing with partner with same bid in non-competitive
	#		Perhaps consider seperating non-competitive from competitive

	#Double only when last bid was opponents
	if length(state.history) > 0 && state.history[end] <= TRUMPBIDS
		push!(bids, DOUBLE)
	elseif length(state.history) > 2 && state.history[end] == PASS && state.history[end-1] == PASS && state.history[end-2] <= TRUMPBIDS

		push!(bids, DOUBLE)
	end

	#Redouble only when opponents doubled
	if length(state.history) > 0 && state.history[end] == DOUBLE
		push!(bids, REDOUBLE)
	elseif length(state.history) > 2 && state.history[end] == PASS && state.history[end-1] == PASS && state.history[end-2] == DOUBLE

		push!(bids, REDOUBLE)
	end

	push!(bids, PASS) #Can always pass
end

#Games end when there is at least 4 bids and last three are PASS
"""
    shouldterminate(state::BridgeState)

Return whether the game should end given the current state.

Criteria is: At least four bids and last three are all Passes.
"""
function shouldterminate(state::BridgeState)
	length(state.history) > 3 && state.history[end] == PASS && state.history[end-1] == PASS && state.history[end-2] == PASS
end

#Figure out whether the current contract is (re)doubled
# Checks last non-pass bid for double/redouble
# Returns: 1=none 2=doubled 4=redoubled
"""
    doublestate(state::BridgeState)

Return an integer representing the doubling state.

`1` for no doubling, `2` for doubling, `4` for redoubling.

See also: [`points`](@ref)
"""
function doublestate(state::BridgeState)
	lastbid = filter(x->x != PASS, state.history)[end]
	if lastbid == REDOUBLE; return 4; end
	if lastbid == DOUBLE; return 2; end
	return 1
end

"""
    bid!(state::BridgeState, bid)

Play a bid and change the game state.

The bid is assumed to be played by the current player and checked for legality.

The game will also terminate if needed and otherwise the current player will be incremented.
"""
function bid!(state::BridgeState, bid)
	@assert bid in legalbids(state)

	push!(state.history, bid)

	if shouldterminate(state)
		state.terminated = true
	else
		nextplayer!(state)
	end
end

"""
    ncparscore(state::BridgeState, refPlayer=NORTH)

Calculate a non-competitive par score for the state and given partnership.

Requires double dummy data to exist.
"""
function ncparscore(state::BridgeState, refPlayer=NORTH)
	bestscore = 0
	for contract in 1:TRUMPBIDS
	    for decl in 1:4
	    	if decl%2 == refPlayer%2
				trump = bidtrump(contract)
				level = bidlevel(contract)
				leader = mod1(decl + 1, 4)
				vul = (iseven(decl) ? state.nsvul : state.ewvul)
				tricks = (iseven(decl) ? state.tricks[leader,trump] : 13-state.tricks[leader,trump])
				pts = points(tricks, trump, level, 0, vul)
				if bestscore < pts
					bestscore = pts
				end
	    	end
	    end
	end
	bestscore
end

"""
    parscore(state::BridgeState)

Calculate par score with caching to avoid re-calculations.

Requires double dummy data to exist.
"""
function parscore(state::BridgeState)
	if state.parscore == nothing
		state.parscore,_,_,_ = parscore(state.tricks, dealer=state.starting_player, nsvul = state.nsvul, ewvul = state.ewvul)
	end
	state.parscore
end


#Generate the final score from the state for NS
"""
    score(state::BridgeState)

Calculate the score for the given state from the perspective of N/S.

Requires double dummy data to exist.
"""
function score(state::BridgeState)
	#@assert state.terminated

	#All passes cancels game, effective score of 0
	if all(x->x==PASS,state.history)
		if state.nonzeropass
			return -parscore(state)
		else
			return 0
		end
	end 

	#All the bookkeeping
	contract = winningbid(state)
	decl = declarer(state, contract)
	trump = bidtrump(contract)
	level = bidlevel(contract)
	leader = mod1(decl + 1, 4)
	vul = (iseven(decl) ? state.nsvul : state.ewvul)
	tricks = (iseven(decl) ? state.tricks[leader,trump] : 13-state.tricks[leader,trump])

	pts = points(tricks, trump, level, doublestate(state), vul)

	#Points from NS perspective, so invert if declarer is EW
	if isodd(decl)
		pts *= -1
	end

	return pts
end

"""

    prettyprint(game::BridgeState, stream=stdout)

Formatted display of the state for human readibility.

Requires double dummy data to exist for some functionality.
"""
function prettyprint(game::BridgeState, stream=stdout)
    prettyprinthands(game, stream)
    prettyprintvulnerable(game, stream)
    println(stream)
    prettyprintauction(game, stream)

    if game.terminated && !isnothing(game.tricks) && !any(isnothing, game.tricks)
    	a_nspts = prettyprintscore(game, stream)

 		prettyprintdda(game, stream)

	    par,bid,decl,iter = parscore(game.tricks, dealer=game.starting_player, nsvul = game.nsvul, ewvul = game.ewvul)
	    println(stream, "\nFinal Contract: $(bidlevelchar(bid))$(bidtrumpchar(bid))")
	    println(stream, "Contract Declarer: $(PLAYERCHAR[decl])")
	    declscore = iseven(decl) ? par : -par
	    println(stream, "Game Score: $(declscore)")

	    println(stream)

	    deltascore = abs(par-a_nspts)
	    println(stream, "Absolute Score difference: $deltascore")
	    println(stream, "Absolute IMP difference: $(imps(deltascore))")
 	end

    println(stream)
end

function __split_hand(cards)
    suits = Dict{Char, Array{Char, 1}}()
    suits['S'] = Array{Char,1}()
    suits['H'] = Array{Char,1}()
    suits['D'] = Array{Char,1}()
    suits['C'] = Array{Char,1}()

    for card in cards
        push!(suits[card[2]], card[1])
    end
    suits
end

function __beautify_hand_suit(suit, cards)
    retstr = ""
    if suit == 'S'
        retstr *= "♠"
    elseif suit == 'H'
        retstr *= "♥"
    elseif suit == 'D'
        retstr *= "♦"
    else
        retstr *= "♣"
    end

    for card in cards
        if card == 'T'
            retstr *= " 10"
        else
            retstr *= " " * card
        end
    end

    retstr
end

function prettyprinthands(game::BridgeState, stream=stdout)
    hands = map(game.hands) do hand
        __split_hand(map(sort(hand, rev=true)) do card
            cardvaluechar(card)*cardsuitchar(card)
        end)
    end

    println(stream, "\tNORTH")
    for suit in "SHDC"
        println(stream, "\t" * __beautify_hand_suit(suit, hands[2][suit]))
    end

    westlen = maximum(map(['S','H','D','C']) do suit; length(__beautify_hand_suit(suit, hands[1][suit])); end)
    #println(stream, westlen)
    println(stream, rpad("WEST",westlen),"\t\t","EAST")
    for suit in "SHDC"
        print(stream, rpad(__beautify_hand_suit(suit, hands[1][suit]), westlen))
        print(stream, "\t\t")
        println(stream, __beautify_hand_suit(suit, hands[3][suit]))
    end

    println(stream, "\tSOUTH")
    for suit in "SHDC"
        println(stream, "\t" * __beautify_hand_suit(suit, hands[4][suit]))
    end
end

function prettyprintdealer(game::BridgeState, stream=stdout)
    print(stream, "Dealer: ")
    if game.starting_player == WEST
        print(stream, "West")
    elseif game.starting_player == NORTH
        print(stream, "North")
    elseif game.starting_player == EAST
        print(stream, "East")
    elseif game.starting_player == SOUTH
        print(stream, "South")
    else
    	print(stream, "What? $game.starting_player")
    end
    println(stream)
end

function prettyprintvulnerable(game::BridgeState, stream=stdout)
    print(stream, "Vulnerability: ")
    if game.nsvul
        if game.ewvul
            print(stream, "Both")
        else
            print(stream, "NS")
        end
    else
        if game.ewvul
            print(stream, "EW")
        else
            print(stream, "None")
        end
    end
    println(stream)
end

function prettyprintauction(game::BridgeState, stream=stdout)
	auction = map(game.history) do bid
        if bid < PASS
            bidlevelchar(bid) * bidtrumpchar(bid)
        elseif bid == PASS
            "Pass"
        elseif bid == DOUBLE
            "X"
        else
            "XX"
        end
    end
    prepend!(auction, fill("-", game.starting_player-1))
    println(stream, "WEST \tNORTH\tEAST \tSOUTH")
    for (i,b) in enumerate(auction)
        print(stream, rpad(b,5),"\t")
        if i%4 == 0 || i==length(auction)
            println(stream)
        end
    end
end

function prettyprintdda(game::BridgeState, stream=stdout)
	println(stream, "\nDouble Dummy Analysis (N/S tricks)")
    println(stream, "   ♣  ♦  ♥  ♠  N")
    for lead in 1:4
        print(stream, PLAYERCHAR[lead])
        for trump in 1:5
            print(stream, " ", lpad(game.tricks[lead,trump], 2))
        end
        println(stream)
    end
end

function prettyprintscore(game::BridgeState, stream=stdout)
	if all(x->x==PASS,game.history) 
	    println(stream, "\nFinal Contract: N/A")
	    println(stream, "Contract Declarer: N/A")
	    println(stream, "Game Score: 0")
	    0
	else
	    a_contract = winningbid(game)
	    a_declarer = declarer(game, a_contract)
	    a_trump = bidtrump(a_contract)
	    a_level = bidlevel(a_contract)
	    a_leader = mod1(a_declarer + 1, 4)
	    a_vul = (iseven(a_declarer) ? game.nsvul : game.ewvul)
	    a_tricks = (iseven(a_declarer) ? game.tricks[a_leader,a_trump] : 13-game.tricks[a_leader,a_trump])
	    a_pts = points(a_tricks, a_trump, a_level, doublestate(game), a_vul)
	    a_nspts = iseven(a_declarer) ? a_pts : -a_pts
	    println(stream, "\nFinal Contract: $(bidlevelchar(a_contract))$(bidtrumpchar(a_contract))")
	    println(stream, "Contract Declarer: $(PLAYERCHAR[a_declarer])")
	    println(stream, "Game Score: $(a_pts)")
	    a_nspts
	end 
end

#=Displays the game which includes
#	All the hands of players
#	All the bid history, if there is at least one bid
#	Final game stats if the game is terminated
function printgame(state::BridgeState)
	println("Hands:")
	for p in 1:4
		print("$(PLAYERCHAR[p]) -")
		for card in state.hands[p]
			print(" $(cardvaluechar(card))$(cardsuitchar(card))")
		end
		println()
	end

	if length(state.history) > 0
		println()
		println("History:")
		curpl = state.starting_player
		for bid in state.history
			print("$(PLAYERCHAR[curpl]): ")
			if bid <= TRUMPBIDS
				print("$(bidtrumpchar(bid))$(bidlevelchar(bid)) ")
			elseif bid == PASS
				print("PA ")
			elseif bid == DOUBLE
				print("DO ")
			elseif bid == REDOUBLE
				print("RE ")
			else
				print("?? ")
			end
			curpl = mod1(curpl + 1 , 4)
		end
		println()
	end

	if state.terminated
		contract = winningbid(state)
		trump = bidtrump(contract)
		level = bidlevel(contract)
		#decl = declarer(state, contract)
		#hands = state.tricks[decl, trump]
		println()
		println("Final contact: $(bidtrumpchar(contract))$(bidlevelchar(contract))")
		#println("Declarer: $(PLAYERCHAR[decl])")
		#println("Tricks won: $(state.tricks[decl, trump])")
		println("Final Reward: $(score(state))")
	end
end=#
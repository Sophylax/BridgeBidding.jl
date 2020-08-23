mutable struct BridgeState
	hands #(Array of four arrays, each having 13 integers for WEST NORTH EAST SOUTH respectively
	history #Array of past bids made until this point
	player #Integer designating the next player
	starting_player #Integer designating the starting player
	nsvul #Boolean for NS vulnerability
	ewvul #Boolean for EW vulnerability
	terminated #Boolean for quick access if the episode ended
	tricks #2D array of number of tricks taken by NS for all trumps and leads, eg. tricks[HEARTS,NORTH]
			#May be nothing if double dummy data is not provided
end

function BridgeState(; starting_player = WEST, nsvul = false, ewvul = false)
	deck = shuffle(1:52)
	hands = [deck[1:13], deck[14:26], deck[27:39], deck[40:52]]
	BridgeState(hands, [], starting_player, starting_player, nsvul, ewvul, false, nothing)
end

function BridgeState(hands; starting_player = WEST, nsvul = false, ewvul = false)
	BridgeState(hands, [], starting_player, starting_player, nsvul, ewvul, false, nothing)
end

function BridgeState(hands, tricks; starting_player = WEST, nsvul = false, ewvul = false)
	BridgeState(hands, [], starting_player, starting_player, nsvul, ewvul, false, tricks)
end

function nextplayer!(state::BridgeState)
	state.player = mod1(state.player + 1, 4)
end

function regularbids(state::BridgeState)
    filter(x -> x <= TRUMPBIDS, state.history)
end

function winningbid(state::BridgeState)
	regulars = regularbids(state)
	if length(regulars) > 0
		return maximum(regulars)
	else
		return 0
	end
end

#Declarer of a bid. Get the bid's maker, find who in that partnership declared that trump
function declarer(state::BridgeState, bid)
	@assert bid in state.history
	maker = findfirst(x -> x==bid, state.history)

	declarer = findfirst(collect(enumerate(state.history))) do (i,x)
		i%2 == maker%2 && bidtrump(x)==bidtrump(bid)
	end
	no_of_player_changes = declarer - 1
	effective_player_changes = no_of_player_changes % 4
	mod1(state.starting_player + effective_player_changes, 4)
end

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
function shouldterminate(state::BridgeState)
	length(state.history) > 3 && state.history[end] == PASS && state.history[end-1] == PASS && state.history[end-2] == PASS
end

#Figure out whether the current contract is (re)doubled
# Checks last non-pass bid for double/redouble
# Returns: 1=none 2=doubled 4=redoubled
function doublestate(state::BridgeState)
	lastbid = filter(x->x != PASS, state.history)[end]
	if lastbid == REDOUBLE; return 4; end
	if lastbid == DOUBLE; return 2; end
	return 1
end

#Play a bid and change the game state
function bid!(state::BridgeState, bid)
	@assert bid in legalbids(state)

	push!(state.history, bid)

	if shouldterminate(state)
		state.terminated = true
	else
		nextplayer!(state)
	end
end

#Generate the final score from the state
#	Returns the difference to parscore for NS
function score(state::BridgeState)
	#@assert state.terminated
	dealer = mod1(state.starting_player - 1, 4)
	par,_,_,_ = parscore(state.tricks, dealer=dealer, nsvul = state.nsvul, ewvul = state.ewvul)

	#All passes cancels game, effective score of 0
	if all(x->x==PASS,state.history) return -par end 

	#All the bookkeeping
	contract = winningbid(state)
	decl = declarer(state, contract)
	trump = bidtrump(contract)
	level = bidlevel(contract)
	leader = mod1(decl + 1, 4)
	vul = (iseven(decl) ? state.nsvul : state.ewvul)
	tricks = (iseven(decl) ? state.tricks[leader,trump] : 13-state.tricks[leader,trump])

	pts = points(tricks, trump, level, doublestate(state), vul)

	#Points from NS perspective
	if isodd(decl)
		pts *= -1
	end

	return pts-par
end

#Displays the game which includes
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
end
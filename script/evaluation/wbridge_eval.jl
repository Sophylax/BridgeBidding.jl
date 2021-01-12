include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras, Statistics, Sockets, StatsBase
import .BridgeBidding: PLAYERCHAR, SUITCHAR, cardvaluechar, cardsuit, PASS, DOUBLE, REDOUBLE, WEST, NORTH, EAST, SOUTH, bid!, bidlevel, bidtrump, legalbids, makebid, winningbid, declarer, getpolicy, score, imps

#Hand vector into the format that Wbridge5 likes
function formathand(hand::Vector{UInt8})
	handstr = ""
	for s in 4:-1:1
	    handstr *= SUITCHAR[s] * " "
	    cards = filter(hand) do card
	    	cardsuit(card) == s
	    end
	    if length(cards) > 0
	    	chars = map(cardvaluechar, cards)
	    	handstr *= join(chars, " ")
	    else
	    	handstr *= "-"
	    end
	    handstr *= ". "
	end
	handstr[1:end-1]
end

#Bid string into our format.
function parsebid(bidstr)
	bidstr = uppercase(bidstr)
	strparts = split(bidstr)
	if strparts[2] == "PASSES"
		return PASS
	elseif strparts[2] == "DOUBLES"
		return DOUBLE
	elseif strparts[2] == "REDOUBLES"
		return REDOUBLE
	elseif strparts[2] == "BIDS"
		bid = strparts[3]
		level = tryparse(Int, string(bid[1]))
		trump = findfirst(x->x==bid[2], SUITCHAR)
		return makebid(level, trump)
	else
		return -1
	end
end

#Our bids into the format Wbridge5 likes.
function writebid(bid)
	if bid == PASS
		return "PASSES"
	elseif bid == DOUBLE
		return "DOUBLES"
	elseif bid == REDOUBLE
		return "REDOUBLES"
	elseif bid < PASS && bid > 0
		bidlev = string(bidlevel(bid))
		bidtru = SUITCHAR[bidtrump(bid)]
		if bidtrump(bid) == 5
			bidtru *= "T"
		end
		return "BIDS " * bidlev * bidtru
	else
		return "??"
	end
end

main(args::AbstractString) = main(split(args))

function main(args::Vector{String})

	greed = true #Hardcoded way to toggle between greedy and stochastic

	ddeval = generate_doubledummy_gameset("../../data/dd-eval-10k.txt", batchsize=1, shuffled=false, mix_games=false)[1];
	model = Knet.load("../../../experiments/ppo3entropy1e-3actor.jld2", "model"); #More hardcoded stuff

	port = 6966 #useful when this runs as multiple instances within the same machine, still hardcoded
	println("Connect to localhost:$port")
	println("Waiting for EAST and WEST")

	server = listen(IPv4(0), port)

	#declaring future variables for sockets
	e_socket = nothing
	w_socket = nothing

	#These two saves both lines and some sanity
	# Ideally I'd monitor what was sent for each message, but most of the messages are just confirmation of state
	readboth = ()->begin
		readline(e_socket)
		readline(w_socket)
	end

	# So many messages need to be sent to both agents
	writeboth = (message)->begin
		write(e_socket, message)
		write(w_socket, message)
	end

	#Attempting to seat
	# First accept a new socket
	# Check the formatting
	# If the requested seat is valid (or no seat is requested), seat and save the socket
	# Break if both east and west is seated
	while true
	    sock = accept(server)
	    response = readline(sock)
	    r_match = match(r"Connecting \"(.*)\" as (\w+) using protocol version (\d+)", response) #Expected introduction message
	    if isnothing(r_match)
	    	println("Invalid Message: $response")
	    else
	    	#Basically, look at the request and attempt to fulfill it.
	    	team = r_match.captures[1]
	    	hand = r_match.captures[2]
	    	if hand == "EAST"
	    		if isnothing(e_socket)
	    			e_socket = sock
	    			write(sock, "EAST \"$team\" seated\r\n")
	    			println("EAST \"$team\" seated")
	    		else
	    			println("East is already seated.")
	    		end
	    	elseif hand == "WEST"
	    		if isnothing(w_socket)
	    			w_socket = sock
	    			write(sock, "WEST \"$team\" seated\r\n")
	    			println("WEST \"$team\" seated")
	    		else
	    			println("West is already seated.")
	    		end
	    	elseif hand == "ANYPL" #Undocumented: Any Player - Wbridge5 will accept all seatings
	    		if isnothing(e_socket)
	    			e_socket = sock
	    			write(sock, "EAST \"$team\" seated\r\n")
	    			println("EAST \"$team\" seated")
	    		else
	    			w_socket = sock
	    			write(sock, "WEST \"$team\" seated\r\n")
	    			println("WEST \"$team\" seated")
	    		end
	    	else
	    		println("Invalid Hand: $hand")
	    	end
	    end

	    if !isnothing(e_socket) && !isnothing(w_socket)
	    	break
	    end
	end

	println("External players are seated, starting runs.")

	readboth()
	writeboth("Teams : N/S : \"DNN\". E/W : \"Wbridge5\"\r\n")
	readboth()

	adv_history = []

	deal_id = 1

	for ddb in take(ddeval, 1000)
		dd = ddb[1] #This is a batch of deals, but we get it in batches of one

		game = BridgeState(dd, starting_player = WEST)

		println("Deal #$deal_id")

		writeboth("Start of Board\r\n")
		readboth()
		writeboth("Board number $((deal_id*2)-1). Dealer WEST. Neither vulnerable\r\n")
		readboth()

		write(w_socket, "WEST's cards : $(formathand(game.hands[WEST]))")
		write(e_socket, "EAST's cards : $(formathand(game.hands[EAST]))")

		#Game Logic
		while !game.terminated
			actor = game.player #This is frozen cuz game.player can change and we need the original actor

			#Wbridge5 Logic
			# Parse bid, update state, relay updates
			if isodd(actor)

				#The only thing different for different seats is the sockets
				# Normally you check the message for the sender, but TCP sockets keep track of that
				# This is mostly the reason of so many repetitions of messages, as the bridge protocol doesn't rely on TCP
				actor_socket = if actor == WEST
					w_socket
				else
					e_socket
				end

				partner_socket = if actor == WEST
					e_socket
				else
					w_socket
				end

				readline(partner_socket) #Boring
				bidtext = readline(actor_socket) #Interesting

				#Passing alerts to partner just generates manual work for the operator
				# So we pretend the alert doesn't exists.
				# It's meant for our network, which doesn't know about alerting.
				# Protocol supports Alert reasoning, but I haven't seen Wbridge5 using the reason part.
				if occursin(" Alert.", bidtext)
					bidtext = replace(bidtext, " Alert." => "")
				end

				action = parsebid(bidtext)
				bid!(game, action)

				#Bidding proceeds, business as usual, relay message to partner
				if !game.terminated 
					write(partner_socket, bidtext)
				#If the bidding is over, we need to juggle some messaging.
				# First of all, we don't relay the bid over to the partner, it will get reset with the next set of messages
				# BUT, if the game is over and the current bidding agent was the leader, then it will immediately go to the playing phase.
				# Meaning it will send an additional message. And we read and discard it to clear the socket.
				# It will also waste some time thinking about the move, but nothing we can do to avoid that.
				else
					if all(game.history .== PASS)

					else
						contract = winningbid(game)
						decl = declarer(game, contract)
						leader = mod1(decl + 1, 4)
						if leader == actor

						else
							readline(actor_socket)
						end
					end
				end
			#Our turn to play
			else
				readboth()

				#Only thing that matters about the actor is giving starting the message with the correct seat.
				# Rest of our logic will sort this by itself already.
				plstr = if actor == NORTH
					"NORTH"
				else
					"SOUTH"
				end

				#Standard bid decision logic
				obs = BridgeBidding.Observation(game)
                policy = Array(softmax(reshape(getpolicy(model, obs), :)))
                legal = legalbids(game)

                action = if greed #There is the greedy switch.
	    			legal[findmax(policy[legal])[2]]
	    		else
	    			sample(legal, Weights(policy[legal]))
	    		end

				msg = "$plstr $(writebid(action))"
				bid!(game, action)

				#Like above, the Wbridge5 doesn't need to know if the game is over.
				if !game.terminated
					writeboth(msg)
				end
			end
		end

		firstscore = score(game)

		#The duplicate game
		# Instead of switching the seats around...
		#  It was overly complicated. We can't actually switch the seats Wbridge5 thinks they are on.
		#  So I tried to switch in our representation and then translate the seats when parsing and formatting the network messages.
		#  A total headache.
		# I switch the game around:
		#  Hands are shifted (W>N>E>S>W)
		#  Double Dummy results are shifted (like above) and inverted (as it represented the tricks taken by NS, so we need to switch)
		#  And the starting player is North instead of West
		game = BridgeState(circshift(dd.hands, 1), 13 .- circshift(dd.tricks, (1,0)), starting_player = NORTH)

		writeboth("Start of Board\r\n")
		readboth()
		writeboth("Board number $((deal_id*2)). Dealer NORTH. Neither vulnerable\r\n")
		readboth()

		#This is a giant copy-paste of above logic
		write(w_socket, "WEST's cards : $(formathand(game.hands[WEST]))")
		write(e_socket, "EAST's cards : $(formathand(game.hands[EAST]))")

		while !game.terminated
			actor = game.player
			if isodd(actor)
				actor_socket = if actor == WEST
					w_socket
				else
					e_socket
				end

				partner_socket = if actor == WEST
					e_socket
				else
					w_socket
				end

				readline(partner_socket)
				bidtext = readline(actor_socket)
				if occursin(" Alert.", bidtext)
					bidtext = replace(bidtext, " Alert." => "")
				end
				action = parsebid(bidtext)
				bid!(game, action)
				if !game.terminated
					write(partner_socket, bidtext)
				else
					if all(game.history .== PASS)

					else
						contract = winningbid(game)
						decl = declarer(game, contract)
						leader = mod1(decl + 1, 4)
						if leader == actor

						else
							readline(actor_socket)
						end
					end
				end
			else
				readboth()
				plstr = if actor == NORTH
					"NORTH"
				else
					"SOUTH"
				end
				obs = BridgeBidding.Observation(game)
                policy = Array(softmax(reshape(getpolicy(model, obs), :)))
                legal = legalbids(game)
                action = if greed
	    			legal[findmax(policy[legal])[2]]
	    		else
	    			sample(legal, Weights(policy[legal]))
	    		end
				msg = "$plstr $(writebid(action))"
				bid!(game, action)
				if !game.terminated
					writeboth(msg)
				end
			end
		end

		secondscore = score(game)

		#Our score (as NS) is the sum of NS scores scaled to Imps
		adv = imps(firstscore + secondscore)
		println("Result: $adv IMPs")

		#Statistics!
		push!(adv_history, adv)
		println("Overall Average: $(mean(adv_history))")
		println("Sample std: $(std(adv_history))")
		println("Average std: $(std(adv_history)/sqrt(length(adv_history)))")

		deal_id += 1
	end

	writeboth("End of session\r\n")

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end

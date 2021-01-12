"""
    BridgeStatesFromPBN(pbnstring::String)

Construct a vector of BridgeState from the given PBN file.
"""
function BridgeStatesFromPBN(pbnstring::String)
    pbnlines = Iterators.Stateful(split(pbnstring, '\n'))

    states = []
    while true
    	state = BridgeStateFromPBN(pbnlines)
        if state == nothing
        	break
        else
        	push!(states, state)
        end
    end
    states
end

function BridgeStateFromPBN(pbnlines)
    hands = nothing
    starting_player = nothing
    nsvul = nothing
    ewvul = nothing

    full = ()->!any(isnothing,(hands,starting_player,ewvul,nsvul))

    for line in pbnlines
    	linematch = match(r"\[(\w+) \\\"(.*)\\\"\]",line)
    	if linematch == nothing
    		continue
    	end

    	tag = linematch[1]
    	content = linematch[2]

    	if tag == "Vulnerable"
    		if content == "Both"
    			nsvul = true
    			ewvul = true
    		elseif content == "None"
    			nsvul = false
    			ewvul = false
    		elseif content == "EW"
    			nsvul = false
    			ewvul = true
    		elseif content == "NS"
    			nsvul = true
    			ewvul = false
    		else
        		error("Invalid vul: $content")
    		end
    	elseif tag == "Dealer"
    		starting_player = findfirst(x->x==content[1], PLAYERCHAR)
    	elseif tag == "Deal"
    		player = findfirst(x->x==content[1], PLAYERCHAR)
    		hands = Array{Any}(fill(nothing, 4))
    		for handstr in split(content[3:end], " ")
    			hand = []
    			for (i, suitstr) in enumerate(split(handstr, "."))
    				for cardchar in suitstr
    					value = findfirst(x->x==cardchar, CARDCHAR)
    					suit = 5-i
    					push!(hand, makecard(suit, value))
    				end
    			end
    			hands[player] = hand
    			player = mod1(player + 1, 4)
    		end
    	end

    	if full()
    		break
    	end
    end

    if full()
    	BridgeState(hands, starting_player=starting_player, nsvul=nsvul, ewvul=ewvul)
    else
    	nothing
    end
end
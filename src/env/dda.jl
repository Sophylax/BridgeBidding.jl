"""
    doubledummyanalysis(state::BridgeState; enginepath::String)

Fill in the double dummy data of given Bridge State, using GIB engine in the given path.  It uses the dd.sh script, written for this function, to interact with GIB. Check the source folder for this function to find the dd.sh. It needs to be placed in the same  folder as the engine executable.
"""
function doubledummyanalysis(state::BridgeState; enginepath::String)
	dealstr = ""
	for p=1:4
		hand = sort(state.hands[p], rev=true)
		dealstr *= PLAYERCHAR[p]
		dealstr *= " "
		for s=4:-1:1
			s_hand = filter(hand) do card cardsuit(card)==s end
			dealstr *= String(Array{Char}(map(s_hand) do card cardvaluechar(card) end))
			if s > 1
				dealstr *= '.'
			end
		end
		dealstr *= "\n"
	end

	cd(()->doubledummyanalysis(state, dealstr),enginepath)
end

function doubledummyanalysis(state::BridgeState, deal::String)
	state.tricks = Array{Any}(fill(nothing, 4, 5))
	@sync for l=1:4
		for t=1:5
			@async doubledummyanalysis(state, deal, l, t)
		end
	end
end

function doubledummyanalysis(state::BridgeState, deal::String, leader::Int, trump::Int)
	run(`mkfifo ddfifo$leader$trump`)
	LTstr = lowercase(deal * PLAYERCHAR[leader] * " " * SUITCHAR[trump] * "\n")
	outpt = read(`bash dd.sh "$LTstr" ddfifo$leader$trump`, String)
	state.tricks[leader, trump] = parse(Int, outpt[28:end-21], base = 16)
	run(`rm ddfifo$leader$trump`)
end
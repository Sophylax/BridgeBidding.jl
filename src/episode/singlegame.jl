"""
	SingleGame(par::Bool, max_steps::Int, dealer, ns_vul, ew_vul)
	SingleGame(par::Bool=false, max_steps::Int=320)

Game format in which a single game is played for each table.

# Fields
- `par::Bool`: If true, the final scores are compared to par scores.
- `max_steps::Int`: Maximum length allowed for bidding sequences.
- `dealer`: Zero argument function returning a player, determining the table dealer on the time of construction.
- `ns_vul`: Zero argument function returning a bool, determining the table N/S vulnerability on the time of construction.
- `ew_vul`: Zero argument function returning a bool, determining the table E/W vulnerability on the time of construction.
"""
struct SingleGame <: GameFormat
	par::Bool
	max_steps::Int
	dealer
	ns_vul
	ew_vul
end

SingleGame(par=false, max_steps=320) = SingleGame(par, max_steps, ()->WEST, ()->false, ()->false)
# Experience: Id, player, Obs, legal, action, probability, next_obs, result

function playbatchedgame(states::Vector{BridgeState}, actor, opponent, max_steps; record_actor = true, record_opponent = true, greedy = false)
	enum = collect(1:length(states))
	logs = [[Experience[] for a in 1:4] for b in 1:length(states)]
	for stp in 1:max_steps
		player = states[enum[1]].player
		model = iseven(player) ? actor : opponent
	    observables = map(states[enum]) do state Observation(state) end
	    policies = Array(softmax(getpolicy(model, observables),dims=1))
	    @sync for (local_id, global_id) in enumerate(enum)
	    	@async begin
	    		legal = legalbids(states[global_id])
	    		action = if greedy
	    			legal[findmax(policies[legal, local_id])[2]]
	    		else
	    			sample(legal, Weights(policies[legal, local_id]))
	    		end
	    		bid!(states[global_id], action)

	    		if (iseven(player) && record_actor) || (isodd(player) && record_opponent)
		    		if stp > 4
		    			logs[global_id][player][end].successor = observables[local_id]
		    		end

		    		push!(logs[global_id][player],
	    				Experience(
	    					global_id,
	    					player,
	    					observables[local_id],
							legal,
							action,
							policies[action,local_id]/sum(policies[legal, local_id]),
							nothing,
							nothing
	    				)
	    			)
		    	end
	    	end
		end
		filter!(enum) do id !states[id].terminated end
		if length(enum) <= 0
	    	break
	    end
	end
	return collect(Iterators.flatten(Iterators.flatten(logs)))
end

function (sGame::SingleGame)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
	states = map(games) do game BridgeState(game, starting_player = sGame.dealer(), nsvul = sGame.ns_vul(), ewvul = sGame.ew_vul(), nonzeropass = nonzeropass) end
	
	logs = playbatchedgame(states, actor, opponent, sGame.max_steps, record_actor=record_actor, record_opponent=record_opponent, greedy=greedy)

	scores = map(states) do state
		if sGame.par
			imps(score(state) - parscore(state)) / 24
		else
			imps(score(state)) / 24
		end
	end

	map(logs) do xp
		if iseven(xp.player)
			xp.result = scores[xp.id]
		else
			xp.result = -scores[xp.id]
		end
	end

	logs, states
end


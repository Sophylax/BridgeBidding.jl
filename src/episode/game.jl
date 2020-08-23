randombaseline(obs) = ones(38)/38

function playepisode!(state::BridgeState; maxsteps::Int=320, agents=fill(randombaseline, 4))
	
	state_actions = []

	for stp in 1:maxsteps
	    player = state.player
	    agent = agents[player]
	    obs = Observation(state)
	    policy = agent(obs)
	    legal = legalbids(state)
	    action = sample(legal, Weights(policy[legal]))

	    bid!(state, action)

	    push!(state_actions, (player, obs, legal, action))

	    if state.terminated
	    	break
	    end
	end
	current_step = 0
	
    state_actions
end

function playepisode(game::DoubleDummy; maxsteps::Int=320, agents=fill(randombaseline, 4))
	state = BridgeState(game)
	playepisode!(state, maxsteps=maxsteps, agents=agents), state
end

function playepisode(games::Array{DoubleDummy,1}; actormodels, starting_player = WEST, maxsteps::Int=320)
	states = map(games) do game BridgeState(game, starting_player=starting_player) end
	enum = collect(1:length(states))
	state_actions = []
	for stp in 1:maxsteps
		player = states[enum[1]].player
		model = actormodels[player]
	    observables = map(states[enum]) do state Observation(state) end
	    policies = Array(softmax(model(observables),dims=1))
	    @sync for (local_id, global_id) in enumerate(enum)
	    	@async begin
	    		legal = legalbids(states[global_id])
	    		action = sample(legal, Weights(policies[legal, local_id]))
	    		bid!(states[global_id], action)
	    		push!(state_actions, (global_id, player, observables[local_id], legal, action))
	    	end
		end
		filter!(enum) do id !states[id].terminated end
		if length(enum) <= 0
	    	break
	    end
	end
	scores = map(states) do state score(state) end
	state_actions = map(state_actions) do (id, player, obs, legal, action)
		if iseven(player)
			return (id, player, obs, legal, action, imps(scores[id])/24)
		else
			return (id, player, obs, legal, action, imps(-scores[id])/24)
		end
	end
	state_actions, states
end
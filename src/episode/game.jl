"""
	abstract type GameFormat

Abstract type representing a way to play bridge. Expected to implement the following callable function:

```
(gFormat::GameFormat)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
```
"""
abstract type GameFormat end

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



function playepisode(games::Array{DoubleDummy,1}; actormodels, starting_player = WEST, maxsteps::Int=320, greedy=fill(false, 4))
	states = map(games) do game BridgeState(game, starting_player=starting_player) end
	enum = collect(1:length(states))
	state_actions = []
	for stp in 1:maxsteps
		player = states[enum[1]].player
		model = actormodels[player]
	    observables = map(states[enum]) do state Observation(state) end
	    policies = Array(softmax(getpolicy(model, observables),dims=1))
	    @sync for (local_id, global_id) in enumerate(enum)
	    	@async begin
	    		legal = legalbids(states[global_id])
	    		action = if greedy[player]
	    			legal[findmax(policies[legal, local_id])[2]]
	    		else
	    			sample(legal, Weights(policies[legal, local_id]))
	    		end
	    		bid!(states[global_id], action)
	    		push!(state_actions, (global_id, player, observables[local_id], legal, action, policies[action,local_id]/sum(policies[legal, local_id]), policies[:,local_id]))
	    	end
		end
		filter!(enum) do id !states[id].terminated end
		if length(enum) <= 0
	    	break
	    end
	end
	scores = map(states) do state score(state)-parscore(state) end
	state_actions = map(state_actions) do (id, player, obs, legal, action, actionprob, policy)
		(
			id=id,
			player=player,
			obs=obs,
			legal=legal,
			action=action,
			score=(imps(scores[id])/24)*(iseven(player) ? 1 : -1),
			actionprob=actionprob,
			policy=policy
		)
	end
	state_actions, states
end
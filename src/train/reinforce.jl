struct Reinforce
	data
	actor
	fixactors::Array{Any,1}
	maxbatch
end

Reinforce(a,d; actors=[nothing,nothing,nothing,nothing], maxbatch=1024) = Reinforce(d,a,actors,maxbatch)
Reinforce!(x...; o...) = for x in Reinforce(x...; o...); end

Base.length(rf::Reinforce) = length(rf.data)
Base.size(rf::Reinforce,d...) = size(rf.data,d...)

function Base.iterate(rf::Reinforce, s...)
    next = iterate(rf.data, s...)
    next === nothing && return nothing
    (games, s) = next

    actors = map(rf.fixactors) do agent
    	if isnothing(agent)
    		rf.actor
    	else
    		agent
    	end
    end

    state_actions,_ = playepisode(games, actormodels=actors, starting_player=rand(1:4))
    #(id, player, obs, legal, action, impscore)
    state_actions = filter(x->isnothing(rf.fixactors[x[2]]), state_actions) #Only take non-fixed actor memories
    sort!(state_actions, by=x->length(x[3].past), rev=true) #Descending sort by history length, for rnn batching

    batchcount = Int(ceil(length(state_actions)/rf.maxbatch))
    batches = map(1:batchcount) do batchid
    	if batchid < batchcount
    		state_actions[(batchid - 1) * rf.maxbatch + 1:batchid*rf.maxbatch]
    	else
    		state_actions[(batchid - 1) * rf.maxbatch + 1:end]
    	end
    end

    #clone = deepcopy(ac.actor)

    actor_losses = map(batches) do batch
    	obs = map(x->x[3], batch)
		legal_id = map(x->x[4], batch)
		illegal_vec = map(legal_id) do legal
			map(1:NUMBIDS) do bid
				if bid in legal
					0
				else
					1
				end
			end
		end
		illegal = atype(cat(illegal_vec..., dims=2))
		action_id = map(x->x[5], batch)
		action_vec = map(action_id) do action
			v = zeros(NUMBIDS)
			v[action] = 1
			v
		end
		action = atype(cat(action_vec..., dims=2))
		rewards = atype(map(x->x[6], batch))

		actor_loss = @diff -PolicyLoss(rf.actor, rewards, obs, illegal, action)
		for param in Knet.params(rf.actor)
			g = grad(actor_loss,param)
			Knet.update!(value(param),g)
		end
		value(actor_loss)
    end

    return (sum(actor_losses),s)
end


function PolicyLoss(model, magnitudes, observations, illegal_matrix, action_matrix)
	scores = model(observations)
	masked = scores .+ (-1e9 * illegal_matrix)
	logprobs = logsoftmax(masked,dims=1)
	logactionprob = reshape(sum(logprobs .* action_matrix, dims=1), :)
	return mean(logactionprob .* magnitudes)
end

#=function REINFORCELoss(model, players, obss, legals, actions)
	logprobs = model(obss)
	logactionprob = map(1:length(actions)) do i
		legal_action = findfirst(x->x==actions[i], legals[i])
		logsoftmax(logprobs[legals[i], i])[legal_action]
	end
	#println(logactionprob)
	side_corrected_losses = map(1:length(players)) do i
		(iseven(players[i]) ? logactionprob[i] : -logactionprob[i])
	end
	return mean(side_corrected_losses)
end

function REINFORCETrainer(model, train_games, opt=Adam())
	agent = x->Array(softmax(reshape(model(x), :)))

    for game in train_games
    	#Play Game
    	state = BridgeState(game.hands, game.tricks) #TODO: Different starting players, different vuls. Random or just go over all??
    	state_actions = playepisode!(state, agents = fill(agent,4))
    	episode_reward = imps(score(state))/24

    	state_actions = reverse(state_actions) #Descending history order for batching
    	players = map(x->x[1], state_actions)
    	obss = map(x->x[2], state_actions)
    	legals = map(x->x[3], state_actions)
    	actions = map(x->x[4], state_actions)
    	J = @diff -episode_reward*REINFORCELoss(model, players, obss, legals, actions)
		for param in Knet.params(model)
			g = grad(J,param)
			Knet.update!(value(param),g)
		end
    end
end

function REINFORCETrainerFixed(varmodel, fixmodel, train_games, opt=Adam())
	varagent = x->Array(softmax(reshape(varmodel(x), :)))
	fixagent = x->Array(softmax(reshape(fixmodel(x), :)))

    for game in train_games
    	#Play Game
    	state = BridgeState(game.hands, game.tricks, starting_player=rand(1:4)) #TODO: Different starting players, different vuls. Random or just go over all??
    	state_actions = playepisode!(state, agents = [fixagent, varagent, fixagent, varagent])
    	episode_reward = imps(score(state))/24

    	state_actions = reverse(state_actions) #Descending history order for batching
    	state_actions = filter(x->iseven(x[1]), state_actions) #Filter NS moves for varmodel
    	players = map(x->x[1], state_actions)
    	obss = map(x->x[2], state_actions)
    	legals = map(x->x[3], state_actions)
    	actions = map(x->x[4], state_actions)
    	J = @diff -episode_reward*REINFORCELoss(varmodel, players, obss, legals, actions)
		for param in Knet.params(varmodel)
			g = grad(J,param)
			Knet.update!(value(param),g)
		end
    end
end=#
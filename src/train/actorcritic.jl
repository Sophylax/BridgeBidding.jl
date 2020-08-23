struct ActorCritic
	data
	actor
	critic
	fixactors::Array{Any,1}
	maxbatch
end

ActorCritic(a,c,d; actors=[nothing,nothing,nothing,nothing], maxbatch=1024) = ActorCritic(d,a,c,actors,maxbatch)
ActorCritic!(x...; o...) = for x in ActorCritic(x...; o...); end

Base.length(ac::ActorCritic) = length(ac.data)
Base.size(ac::ActorCritic,d...) = size(ac.data,d...)

function Base.iterate(ac::ActorCritic, s...)
    next = iterate(ac.data, s...)
    next === nothing && return nothing
    (games, s) = next

    actors = map(ac.fixactors) do agent
    	if isnothing(agent)
    		ac.actor
    	else
    		agent
    	end
    end

    state_actions,_ = playepisode(games, actormodels=actors, starting_player=rand(1:4))
    #(id, player, obs, legal, action, impscore)
    state_actions = filter(x->isnothing(ac.fixactors[x[2]]), state_actions) #Only take non-fixed actor memories
    sort!(state_actions, by=x->length(x[3].past), rev=true) #Descending sort by history length, for rnn batching

    batchcount = Int(ceil(length(state_actions)/ac.maxbatch))
    batches = map(1:batchcount) do batchid
    	if batchid < batchcount
    		state_actions[(batchid - 1) * ac.maxbatch + 1:batchid*ac.maxbatch]
    	else
    		state_actions[(batchid - 1) * ac.maxbatch + 1:end]
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
		reward = map(x->x[6], batch)

	    baseline = ac.critic(obs)
	    advantages = atype(reward) .- baseline

		actor_loss = @diff -PolicyLoss(ac.actor, advantages, obs, illegal, action)
		for param in Knet.params(ac.actor)
			g = grad(actor_loss,param)
			Knet.update!(value(param),g)
		end
		value(actor_loss)
    end

    critic_losses = map(batches) do batch
    	obs = map(x->x[3], batch)
		reward = map(x->x[6], batch)

		critic_loss = @diff mean(abs2.(atype(reward) .- ac.critic(obs)))
		for param in Knet.params(ac.critic)
			g = grad(critic_loss,param)
			Knet.update!(value(param),g)
		end
		value(critic_loss)
    end

    return ((sum(actor_losses),sum(critic_losses)),s)
end

#=function ActorLoss_old(model, advantages, obss, legals, actions)
	logprobs = model(obss)
	logactionprob = map(1:length(actions)) do i
		legal_action = findfirst(x->x==actions[i], legals[i])
		logsoftmax(logprobs[legals[i], i])[legal_action]
	end
	return mean(logactionprob .* advantages)
end

function ActorCriticTrainer(actormodel, criticmodel, train_games, opt=Adam())
	agent = x->Array(softmax(reshape(actormodel(x), :)))

    for (i,game) in enumerate(train_games)
    	print("\r$i / $(length(train_games))")
    	#Play Game
    	state = BridgeState(game.hands, game.tricks) #TODO: Different starting players, different vuls. Random or just go over all??
    	state_actions = playepisode!(state, agents = fill(agent,4))
    	episode_reward = imps(score(state))/24

    	state_actions = reverse(state_actions) #Descending history order for batching
    	players = map(x->x[1], state_actions)
    	rewards = map(x->iseven(x) ? episode_reward : -episode_reward, players)
    	#println(rewards)
    	obss = map(x->x[2], state_actions)
    	val_preds = criticmodel(obss)
    	#println(val_preds)
    	advantages = rewards .- Array(val_preds)
    	#println(advantages)
    	legals = map(x->x[3], state_actions)
    	actions = map(x->x[4], state_actions)
    	aJ = @diff -ActorLoss(actormodel, advantages, obss, legals, actions)
		for param in Knet.params(actormodel)
			g = grad(aJ,param)
			Knet.update!(value(param),g)
		end
		cJ = @diff mean(abs2.(atype(rewards) .- criticmodel(obss)))
		for param in Knet.params(criticmodel)
			g = grad(cJ,param)
			Knet.update!(value(param),g)
		end
    end
    println()
end

function ActorCriticTrainerFixed(varactor, critic, fixedactor, train_games, opt=Adam())
	varagent = x->Array(softmax(reshape(varactor(x), :)))
	fixagent = x->Array(softmax(reshape(fixedactor(x), :)))

    for (i,game) in enumerate(train_games)
    	print("\r$i / $(length(train_games))")
    	#Play Game
    	state = BridgeState(game.hands, game.tricks, starting_player=rand(1:4)) #TODO: Different starting players, different vuls. Random or just go over all??
    	state_actions = playepisode!(state, agents = [fixagent, varagent, fixagent, varagent])
    	episode_reward = imps(score(state))/24

    	state_actions = reverse(state_actions) #Descending history order for batching
    	state_actions = filter(x->iseven(x[1]), state_actions) #Filter NS moves for varactor
    	players = map(x->x[1], state_actions)
    	rewards = map(x->iseven(x) ? episode_reward : -episode_reward, players)
    	obss = map(x->x[2], state_actions)
    	val_preds = critic(obss)
    	advantages = rewards .- Array(val_preds)
    	legals = map(x->x[3], state_actions)
    	actions = map(x->x[4], state_actions)
    	aJ = @diff -ActorLoss(varactor, advantages, obss, legals, actions)
		for param in Knet.params(varactor)
			g = grad(aJ,param)
			Knet.update!(value(param),g)
		end
		cJ = @diff mean(abs2.(atype(rewards) .- critic(obss)))
		for param in Knet.params(critic)
			g = grad(cJ,param)
			Knet.update!(value(param),g)
		end
    end
    println()
end=#
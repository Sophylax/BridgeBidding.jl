"""
	Experience(...)

A single point of experience, generated in reinforcement learning environment.

# Fields
- `id::Int`: Identification for the game (inside the batch) that generated this experience.
- `player::Int`: Which seat generated this experience
- `observation`: Their observation at the time of the experience
- `legal`: Array of legal bids at that state.
- `action`: The action the agent made.
- `probability`: The probability the agent assigned to the action.
- `successor`: The next observable the agent got. Might be nothing if it was the last.
- `result`: The final score for the game.
"""
mutable struct Experience
	id
	player
	observation
	legal
	action
	probability
	successor
	result
end

"""
	ExperienceBatch(experiences::Vector{Experience}) 

A collection experiences. It wraps following properties, so that they are extracted from the underlying collection and batched together.

# Properties
- `observations`
- `legals`
- `illegals`
- `actions`
- `probabilities`
- `successors`
- `results`
"""
mutable struct ExperienceBatch
	experiences::Vector{Experience}
	_observations
	_legals
	_illegals
	_actions
	_probabilities
	_successors
	_results
end

ExperienceBatch(experiences::Vector{Experience}) = ExperienceBatch(experiences, nothing, nothing, nothing, nothing, nothing, nothing, nothing)

function Base.getproperty(ebatch::ExperienceBatch, v::Symbol)
    if v == :observations
        return getobservations(ebatch)
    elseif v == :legals
        return getlegals(ebatch)
    elseif v == :illegals
        return getillegals(ebatch)
    elseif v == :actions
        return getactions(ebatch)
    elseif v == :probabilities
        return getprobabilities(ebatch)
    elseif v == :successors
        return getsuccessors(ebatch)
    elseif v == :results
        return getresults(ebatch)
    else
        return getfield(ebatch, v)
    end
end

function getobservations(ebatch::ExperienceBatch)
	if ebatch._observations == nothing
		ebatch._observations = asyncmap(x->x.observation, ebatch.experiences)
	end
	return ebatch._observations    
end

function getlegals(ebatch::ExperienceBatch)
	if ebatch._legals == nothing
		legal_id = asyncmap(x->x.legal, ebatch.experiences)
		legal_vec = asyncmap(legal_id) do legal
			v = zeros(NUMBIDS)
			v[legal] .= 1
			v
		end
		ebatch._legals = cat(legal_vec..., dims=2)
		ebatch._illegals = 1 .- ebatch._legals
	end
	return ebatch._legals    
end

function getillegals(ebatch::ExperienceBatch)
	if ebatch._illegals == nothing
		legal_id = asyncmap(x->x.legal, ebatch.experiences)
		illegal_vec = asyncmap(legal_id) do legal
			v = ones(NUMBIDS)
			v[legal] .= 0
			v
		end
		ebatch._illegals = cat(illegal_vec..., dims=2)
		ebatch._legals = 1 .- ebatch._illegals
	end
	return ebatch._illegals    
end

function getactions(ebatch::ExperienceBatch)
	if ebatch._actions == nothing
		action_id = asyncmap(x->x.action, ebatch.experiences)
		action_vec = asyncmap(action_id) do action
			v = zeros(NUMBIDS)
			v[action] = 1
			v
		end
		ebatch._actions = cat(action_vec..., dims=2)
	end
	return ebatch._actions    
end

function getprobabilities(ebatch::ExperienceBatch)
	if ebatch._probabilities == nothing
        ebatch._probabilities = asyncmap(x->x.probability, ebatch.experiences)
	end
	return ebatch._probabilities    
end

function getsuccessors(ebatch::ExperienceBatch)
	if ebatch._successors == nothing
		ebatch._successors = asyncmap(x->x.successor, ebatch.experiences)
	end
	return ebatch._successors    
end

function getresults(ebatch::ExperienceBatch)
	if ebatch._results == nothing
		ebatch._results = asyncmap(x->x.result, ebatch.experiences)
	end
	return ebatch._results    
end
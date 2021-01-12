function PolicyLoss(model, magnitudes, observations, illegal_matrix, action_matrix; entropy_weight=0)
	scores = getpolicy(model, observations)
	masked = scores .+ (-1e9 * illegal_matrix)
	logprobs = logsoftmax(masked, dims=1)
	e_probs = exp.(logprobs) .* entropy_weight
	entropy = reshape(sum(e_probs .* logprobs, dims=1), :)
	logactionprob = reshape(sum(logprobs .* action_matrix, dims=1), :)
	return -mean((logactionprob .* magnitudes) + entropy)
end

function PPOLoss(model, magnitudes, observations, illegal_matrix, action_matrix, old_probs, margin; entropy_weight=0)
    scores = getpolicy(model, observations)
    masked = scores .+ (-1e9 * illegal_matrix)

    logprobs = logsoftmax(masked, dims=1)
    total_probs = softmax(masked, dims=1)
    action_probs = reshape(sum(total_probs .* action_matrix, dims=1), :)

    prob_ratio = action_probs ./ old_probs
    policy_loss = prob_ratio .* magnitudes

    signs = 1 .- (2 .* (magnitudes .< 0))
    clips = (1 .+ (margin .* signs)) .* magnitudes

    clipped_loss = min.(policy_loss, clips)
    entropy_loss = entropy_weight .* reshape(sum(total_probs .* logprobs, dims=1), :)

    return -mean(clipped_loss + entropy_loss)
end

struct ActorLoss
	reward_surrogate
end

function ValuePrediction(model, batch::ExperienceBatch)
	getvalue(model, batch.observations)
end

function FinalMinusValue(model, batch::ExperienceBatch)
	baseline = ValuePrediction(model, batch)
	atype(batch.results) .- baseline
end

function OneStepValueDifference(model, batch::ExperienceBatch)
	baseline = ValuePrediction(model, batch)

	successors = batch.successors
	terminates = isnothing.(successors)
	intermediates = map(!, terminates)
	new_states = Array{Observation}(successors[intermediates])

	targets = atype(batch.results)
	predicts = Array(value(getvalue(model, new_states)))
	targets[intermediates] .= predicts

	targets .- baseline
end

function (aloss::ActorLoss)(model, batch::ExperienceBatch; entropy_weight=0)
	reward = value(aloss.reward_surrogate(model, batch))
	PolicyLoss(model, reward, batch.observations, atype(batch.illegals), atype(batch.actions), entropy_weight=entropy_weight)
end

function MonteCarloLoss(model, batch::ExperienceBatch)
	mean(abs2.(FinalMinusValue(model, batch)))
end

function TDZeroLoss(model, batch::ExperienceBatch)
	mean(abs2.(OneStepValueDifference(model, batch)))
end

struct ProximalLoss
	reward_surrogate
	margin
end

ProximalLoss(reward_surrogate) = ProximalLoss(reward_surrogate, 0.2)

function (ploss::ProximalLoss)(model, batch::ExperienceBatch; entropy_weight=0)
	reward = value(ploss.reward_surrogate(model, batch))
	PPOLoss(model, reward, batch.observations, atype(batch.illegals), atype(batch.actions), atype(batch.probabilities), ploss.margin, entropy_weight=entropy_weight)
end

struct PositiveContractBidAmplifier
	reward_surrogate
	magnitude
end

PositiveContractBidAmplifier(reward_surrogate) = PositiveContractBidAmplifier(reward_surrogate, 5)

function (amp::PositiveContractBidAmplifier)(model, batch::ExperienceBatch)
	reward = amp.reward_surrogate(model, batch)
	pos_scores = atype(batch.results) .> 0
	contract_bids = atype(asyncmap(x->x.action, batch.experiences)) .< PASS
	and_mask = pos_scores .* contract_bids

	amplifier = (and_mask .* (amp.magnitude - 1)) .+ 1

	reward .* amplifier
end
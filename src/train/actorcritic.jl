"""
    ActorCritic(model, dataset; kwargs...)

A generic Actor-Critic structure for policy gradient methods.

# Keyword Arguments
- `opponent::OpponentScheduler`: Scheduler that decides on the opponents during training.
- `gameformat::GameFormat`: Defines the GameFormat that generates the experiences during training. 
- `actorloss`: A function which takes the model and the experience batch and returns the loss for the actor.
- `criticloss`: A function which takes the model and the experience batch and returns the loss for the critic.
- `entropy`: Entropy weight, passed to the actor loss as the keyword argument `entropy_weight`.
- `maxbatch::Int`: The maximum batchsize for the experience batches.
- `epochs::Int`: How many times the experiences will be used to perform gradient descent before next set of games are played.
- `passscheduler::AnyScheduler`: Scheduler that decides what is being passed to the `nonzeropass` kwarg of the gameformat.
"""
struct ActorCritic
	data
    model::Union{BridgeMultiModel, BridgeMergedModel}
	opponent::OpponentScheduler
    gameformat::GameFormat
    actorloss
    criticloss
    entropy
    maxbatch::Int
    epochs::Int
    passscheduler::AnyScheduler
end

function ActorCritic(m,d; opponent=SelfScheduler(), gameformat=DuplicateGame(),
                    actorloss = ActorLoss(FinalMinusValue), criticloss = MonteCarloLoss,
                    entropy = 0, maxbatch::Int = 512, epochs::Int = 1, passscheduler = AnyScheduler([(1,true)]))
    ActorCritic(d,m,opponent,gameformat,actorloss,criticloss,entropy,maxbatch,epochs, passscheduler)
end
ActorCritic!(x...; o...) = for x in ActorCritic(x...; o...); end

Base.length(ac::ActorCritic) = length(ac.data)
Base.size(ac::ActorCritic,d...) = size(ac.data,d...)

global scorestats = []

function Base.iterate(ac::ActorCritic, s...)
    next = iterate(ac.data, s...)
    next === nothing && return nothing
    (games, s) = next

    opponent = ac.opponent(ac.model)
    experiences,sts = ac.gameformat(games, actor=ac.model, opponent=opponent, nonzeropass=ac.passscheduler())
    sort!(experiences, by=xp->length(xp.observation.past), rev=true) #Descending sort by history length, for rnn batching

    losses = map(ncycle(Iterators.partition(experiences, ac.maxbatch), ac.epochs)) do batch
        exp_batch = ExperienceBatch(collect(Experience, batch))

        loss = @diff ac.actorloss(ac.model, exp_batch, entropy_weight = ac.entropy) + ac.criticloss(ac.model, exp_batch)
        for param in Knet.params(ac.model)
            g = grad(loss, param)
            if !isnothing(g)
                Knet.update!(param,g)
            end
        end
        value(loss)*length(batch)
    end

    return (sum(losses)/length(experiences),s)
end

"""
    A2C(model, dataset; kwargs...)

Trainer iterator for Advantage Actor Critic. As `ActorCritic`, but `actorloss`, `criticloss`, and `epochs` are predefined.

See also: [`ActorCritic`](@ref)
"""
function A2C(m,d; opponent=SelfScheduler(), gameformat=DuplicateGame(),
                    entropy = 0, maxbatch::Int = 512, passscheduler = AnyScheduler([(1,true)]))
    ActorCritic(d,m,opponent,gameformat,ActorLoss(FinalMinusValue),MonteCarloLoss,entropy,maxbatch,1,passscheduler)
end
A2C!(x...; o...) = for x in A2C(x...; o...); end

"""
    ProximalPolicyOptimization(model, dataset; kwargs...)

Trainer iterator for Proximal Policy Optimization. As `ActorCritic`, but `actorloss`, `criticloss` are predefined.

See also: [`ActorCritic`](@ref)
"""
function ProximalPolicyOptimization(m,d; opponent=SelfScheduler(), gameformat=DuplicateGame(),
                    entropy = 0, maxbatch::Int = 512, epochs::Int = 1, passscheduler = AnyScheduler([(1,true)]))
    ActorCritic(d,m,opponent,gameformat,ProximalLoss(FinalMinusValue),MonteCarloLoss,entropy,maxbatch,epochs,passscheduler)
end
ProximalPolicyOptimization!(x...; o...) = for x in ProximalPolicyOptimization(x...; o...); end

"""
    REINFORCE(model, dataset; kwargs...)

Trainer iterator for REINFORCE. As `ActorCritic`, but `actorloss`, `criticloss`, and `epochs` are predefined.

See also: [`ActorCritic`](@ref)
"""
function REINFORCE(m,d; opponent=SelfScheduler(), gameformat=DuplicateGame(),
                    entropy = 0, maxbatch::Int = 512, passscheduler = AnyScheduler([(1,true)]))
    ActorCritic(d,m,opponent,gameformat,ActorLoss((m,b)->b.results),(m,b)->0,entropy,maxbatch,1,passscheduler)
end
REINFORCE!(x...; o...) = for x in REINFORCE(x...; o...); end
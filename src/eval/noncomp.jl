"""
    NonCompEvaluator(dataset; onnewbest = nothing)

Evaluator that plays a game non-competitively and records the distance to the non-competitive par score.

Optionally with *onnewbest*, it will track the best performance and call *onnewbest()* when a new best is reached.

    (nce::NonCompEvaluator)(model)

Run the evaluation using the given model.
"""
mutable struct NonCompEvaluator
    dataset
    lastbest
    onnewbest
end

function NonCompEvaluator(dataset; onnewbest = nothing)
    NonCompEvaluator(dataset, nothing, onnewbest)
end

function (nce::NonCompEvaluator)(model)
    total_score = 0
    game_count = 0
    g_format = SingleGame()
    for games in nce.dataset
        #Play Game
        _,states = g_format(games, actor=model, opponent=AllPassModel(), record_actor=false, greedy=true)
        ncpars = ncparscore.(states)
        scores = score.(states)
        deltaimps = imps.(ncpars .- scores)

        total_score += sum(deltaimps)  #TODO: Different starting players, different vuls. Random or just go over all??
        game_count += length(states)
    end
    game_avg = total_score / game_count
    @info "Non-Comp Cost: $(game_avg)"

    if !isnothing(nce.onnewbest)
        if isnothing(nce.lastbest)
            nce.lastbest = game_avg
        elseif nce.lastbest > game_avg
            nce.lastbest = game_avg
            nce.onnewbest()
        end
    end
    (:score, game_avg)
end
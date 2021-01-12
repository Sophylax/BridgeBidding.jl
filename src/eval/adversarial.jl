"""
    AdversarialEvaluator(dataset; onnewbest = nothing)

Evaluator that plays a duplicate game versus given opponent and records the relative performance.

Optionally with *onnewbest*, it will track the best performance and call *onnewbest()* when a new best is reached.

    (ae::AdversarialEvaluator)(model1, model2; verbose=true, greedy1=true, greedy2=true)

Run the evaluation using model1 and model2, result is from model1's perspective. Verbose enables info prints, greedy1/2 enables greedy policy.
"""
mutable struct AdversarialEvaluator
    dataset
    lastbest
    onnewbest
end

function AdversarialEvaluator(dataset; onnewbest = nothing)
    AdversarialEvaluator(dataset, nothing, onnewbest)
end

function (ae::AdversarialEvaluator)(model1, model2; verbose=true, greedy1=true, greedy2=true)
    total_score = 0
    game_count = 0
    for games in ae.dataset
        #Play Game
        _,ns_states = playepisode(games, actormodels=[model2, model1, model2, model1], greedy=[greedy2, greedy1, greedy2, greedy1])  #TODO: Different starting players, different vuls. Random or just go over all??
        positive_scores = map(ns_states) do st score(st) end

        _,ew_states = playepisode(games, actormodels=[model1, model2, model1, model2], greedy=[greedy1, greedy2, greedy1, greedy2])  #TODO: Different starting players, different vuls. Random or just go over all??
        negative_scores = map(ew_states) do st score(st) end

        total_score += sum(imps.(positive_scores .- negative_scores))
        game_count += length(games)
    end
    game_avg = total_score / game_count
    if verbose
        @info "Adversarial Imp Advantage: $(game_avg)"
    end

    if !isnothing(ae.onnewbest)
        if isnothing(ae.lastbest)
            ae.lastbest = game_avg
        elseif ae.lastbest < game_avg
            ae.lastbest = game_avg
            ae.onnewbest()
        end
    end
    (:game_avg, game_avg)
end
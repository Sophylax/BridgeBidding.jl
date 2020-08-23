mutable struct AdversarialEvaluator
    dataset
    lastbest
    onnewbest
end

function AdversarialEvaluator(dataset; onnewbest = nothing)
    AdversarialEvaluator(dataset, nothing, onnewbest)
end

function (ae::AdversarialEvaluator)(model1, model2)
    println(stderr)

    total_score = 0
    game_count = 0
    for games in ae.dataset
        #Play Game
        _,ns_states = playepisode(games, actormodels=[model2, model1, model2, model1])  #TODO: Different starting players, different vuls. Random or just go over all??
        positive_scores = map(ns_states) do st score(st) end

        _,ew_states = playepisode(games, actormodels=[model1, model2, model1, model2])  #TODO: Different starting players, different vuls. Random or just go over all??
        negative_scores = map(ew_states) do st score(st) end

        total_score += sum(imps.(positive_scores .- negative_scores))
        game_count += length(games)
    end
    game_avg = total_score / game_count
    @info "Adversarial Imp Advantage: $(game_avg)"

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
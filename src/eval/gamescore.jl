"""
    GameScoreEvaluator(dataset; onnewbest = nothing)

Evaluator that plays a game versus itself and records the distance to the par score.

Optionally with *onnewbest*, it will track the best performance and call *onnewbest()* when a new best is reached.

    (gse::GameScoreEvaluator)(model)

Run the evaluation using the given model.
"""
mutable struct GameScoreEvaluator
    dataset
    lastbest
    onnewbest
end

function GameScoreEvaluator(dataset; onnewbest = nothing)
    GameScoreEvaluator(dataset, nothing, onnewbest)
end

function (gse::GameScoreEvaluator)(model)
    total_score = 0
    game_count = 0
    for games in gse.dataset
        #Play Game
        _,states = playepisode(games, actormodels=fill(model, 4), greedy=fill(true, 4))
        total_score += sum(map(states) do st imps(abs(score(st)-parscore(st))) end)  #TODO: Different starting players, different vuls. Random or just go over all??
        game_count += length(states)
    end
    game_avg = total_score / game_count
    @info "Absolute Imp Score: $(game_avg)"

    if !isnothing(gse.onnewbest)
        if isnothing(gse.lastbest)
            gse.lastbest = game_avg
        elseif gse.lastbest > game_avg
            gse.lastbest = game_avg
            gse.onnewbest()
        end
    end
    (:score, game_avg)
end
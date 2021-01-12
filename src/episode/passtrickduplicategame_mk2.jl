"""
	PassTrickDuplicateGame(max_steps::Int, dealer, ns_vul, ew_vul, variant::Bool)
	PassTrickDuplicateGame(max_steps::Int=320, variant = false)

Variation over DuplicateGame, in which reward symmetry is broken when any table all pass.

The new rewards are the negative of par score for that deal, independent of the performance of the other table.

# Fields
- `variant::Bool`: If true the non-all-pass game is also not compared. Otherwise it is compared to the par score.

See also: [`DuplicateGame`](@ref)
"""
struct PassTrickDuplicateGameMkII <: GameFormat
	max_steps::Int
	dealer
	ns_vul
	ew_vul
	variant::Bool
end

PassTrickDuplicateGameMkII(max_steps=320; variant = false) = PassTrickDuplicateGameMkII(max_steps, ()->WEST, ()->false, ()->false, variant)
# Experience: Id, player, Obs, legal, action, probability, next_obs, result

function (dGame::PassTrickDuplicateGameMkII)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
	p_states = map(games) do game BridgeState(game, starting_player = dGame.dealer(), nsvul = dGame.ns_vul(), ewvul = dGame.ew_vul()) end
	s_states = deepcopy.(p_states)

	p_logs = playbatchedgame(p_states, actor, opponent, dGame.max_steps, record_actor=record_actor, record_opponent=record_opponent, greedy=greedy)
	s_logs = playbatchedgame(s_states, opponent, actor, dGame.max_steps, record_actor=record_opponent, record_opponent=record_actor, greedy=greedy)

	p_passes = map(1:length(p_states)) do i
		all(p_states[i].history .== PASS)
	end

	s_passes = map(1:length(p_states)) do i
		all(s_states[i].history .== PASS)
	end

	scores = map(1:length(p_states)) do i
		imps(score(p_states[i]) - score(s_states[i])) / 24
	end

	map(p_logs) do xp
		ns_result = 0
		if p_passes[xp.id]
			ns_result = -imps(parscore(p_states[xp.id])) / 24
		elseif s_passes[xp.id]
			if dGame.variant
				ns_result = imps(score(p_states[xp.id])) / 24
			else
				ns_result = imps(score(p_states[xp.id]) - parscore(p_states[xp.id])) / 24
			end
		else
			ns_result = scores[xp.id]
		end

		if iseven(xp.player)
			xp.result = ns_result
		else
			xp.result = -ns_result
		end
	end

	map(s_logs) do xp
		ew_result = 0
		if s_passes[xp.id]
			ew_result = -imps(parscore(s_states[xp.id])) / 24
		elseif p_passes[xp.id]
			if dGame.variant
				ew_result = imps(score(s_states[xp.id])) / 24
			else
				ew_result = imps(score(s_states[xp.id]) - parscore(s_states[xp.id])) / 24
			end
		else
			ew_result = scores[xp.id]
		end

		if isodd(xp.player)
			xp.result = ew_result
		else
			xp.result = -ew_result
		end
	end

	cat(p_logs, s_logs, dims=1), cat(p_states, s_states, dims=1)

end


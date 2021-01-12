"""
	PassTrickDuplicateGame(max_steps::Int, dealer, ns_vul, ew_vul)
	PassTrickDuplicateGame(max_steps::Int=320)

Variation over DuplicateGame, in which reward symmetry is broken when both tables all pass.

The new rewards are the negative of par score for that deal, independent of the performance of the other table.

See also: [`DuplicateGame`](@ref)
"""
struct PassTrickDuplicateGame <: GameFormat
	max_steps::Int
	dealer
	ns_vul
	ew_vul
end

PassTrickDuplicateGame(max_steps=320) = PassTrickDuplicateGame(max_steps, ()->WEST, ()->false, ()->false)
# Experience: Id, player, Obs, legal, action, probability, next_obs, result

function (dGame::PassTrickDuplicateGame)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
	p_states = map(games) do game BridgeState(game, starting_player = dGame.dealer(), nsvul = dGame.ns_vul(), ewvul = dGame.ew_vul(), nonzeropass = nonzeropass) end
	s_states = deepcopy.(p_states)

	p_logs = playbatchedgame(p_states, actor, opponent, dGame.max_steps, record_actor=record_actor, record_opponent=record_opponent, greedy=greedy)
	s_logs = playbatchedgame(s_states, opponent, actor, dGame.max_steps, record_actor=record_opponent, record_opponent=record_actor, greedy=greedy)

	allpasses = map(1:length(p_states)) do i
		all(p_states[i].history .== PASS) && all(s_states[i].history .== PASS)
	end

	scores = map(1:length(p_states)) do i
		imps(score(p_states[i]) - score(s_states[i])) / 24
	end

	map(p_logs) do xp
		if allpasses[xp.id]
			if iseven(xp.player)
				xp.result = -imps(parscore(p_states[xp.id])) / 24
			else
				xp.result = imps(parscore(p_states[xp.id])) / 24
			end
		else
			if iseven(xp.player)
				xp.result = scores[xp.id]
			else
				xp.result = -scores[xp.id]
			end
		end
	end

	map(s_logs) do xp
		if allpasses[xp.id]
			if isodd(xp.player)
				xp.result = -imps(parscore(s_states[xp.id])) / 24
			else
				xp.result = imps(parscore(s_states[xp.id])) / 24
			end
		else
			if isodd(xp.player)
				xp.result = scores[xp.id]
			else
				xp.result = -scores[xp.id]
			end
		end
		xp.id += length(games)
	end

	cat(p_logs, s_logs, dims=1), cat(p_states, s_states, dims=1)

end


"""
	DuplicateGame(max_steps::Int, dealer, ns_vul, ew_vul)
	DuplicateGame(max_steps::Int=320)

Game format in which a duplicate game is played for each table.

# Fields
- `max_steps::Int`: Maximum length allowed for bidding sequences.
- `dealer`: Zero argument function returning a player, determining the table dealer on the time of construction.
- `ns_vul`: Zero argument function returning a bool, determining the table N/S vulnerability on the time of construction.
- `ew_vul`: Zero argument function returning a bool, determining the table E/W vulnerability on the time of construction.
"""
struct DuplicateGame <: GameFormat
	max_steps::Int
	dealer
	ns_vul
	ew_vul
end

DuplicateGame(max_steps=320) = DuplicateGame(max_steps, ()->WEST, ()->false, ()->false)
# Experience: Id, player, Obs, legal, action, probability, next_obs, result

function (dGame::DuplicateGame)(games::Array{DoubleDummy,1}; actor, opponent, record_actor = true, record_opponent = false, greedy = false, nonzeropass = false)
	p_states = map(games) do game BridgeState(game, starting_player = dGame.dealer(), nsvul = dGame.ns_vul(), ewvul = dGame.ew_vul(), nonzeropass = nonzeropass) end
	s_states = deepcopy.(p_states)

	p_logs = playbatchedgame(p_states, actor, opponent, dGame.max_steps, record_actor=record_actor, record_opponent=record_opponent, greedy=greedy)
	s_logs = playbatchedgame(s_states, opponent, actor, dGame.max_steps, record_actor=record_opponent, record_opponent=record_actor, greedy=greedy)

	scores = map(1:length(p_states)) do i
		imps(score(p_states[i]) - score(s_states[i])) / 24
	end

	map(p_logs) do xp
		if iseven(xp.player)
			xp.result = scores[xp.id]
		else
			xp.result = -scores[xp.id]
		end
	end

	map(s_logs) do xp
		if isodd(xp.player)
			xp.result = scores[xp.id]
		else
			xp.result = -scores[xp.id]
		end
		xp.id += length(games)
	end

	cat(p_logs, s_logs, dims=1), cat(p_states, s_states, dims=1)

end


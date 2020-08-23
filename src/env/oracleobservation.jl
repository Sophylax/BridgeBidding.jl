struct OracleObservation
    hand::Array{Int}
    handleft::Array{Int}
    handpartner::Array{Int}
    handright::Array{Int}
    past::Array{Int}
    vul::Tuple{Int, Int}
end

#Get observation from bridge state for the current player
function OracleObservation(state::BridgeState)
    h1 = state.hands[state.player]
    h2 = state.hands[mod1(state.player + 1, 4)]
    h3 = state.hands[mod1(state.player + 2, 4)]
    h4 = state.hands[mod1(state.player + 3, 4)]
    p = state.history
    if isodd(state.player)
    	return Observation(h1,h2,h3,h4,p,(Int(state.ewvul), Int(state.nsvul)))
    else
    	return Observation(h1,h2,h3,h4,p,(Int(state.nsvul), Int(state.ewvul)))

    end
end
"""
    Observation(hand::Array{Int}, past::Array{Int}, vul::Tuple{Int, Int})

The standard structure representing a player's perspective in auction.

It can be called directly by giving BridgeState, which gives the Observation for the current player. 
"""
struct Observation
    hand::Array{Int}
    past::Array{Int}
    vul::Tuple{Int, Int}
end

#Get observation from bridge state for the current player
function Observation(state::BridgeState)
    h = state.hands[state.player]
    p = state.history
    if isodd(state.player)
    	return Observation(h,p,(Int(state.ewvul), Int(state.nsvul)))
    else
    	return Observation(h,p,(Int(state.nsvul), Int(state.ewvul)))

    end
end
struct MaskedHistoryObserver <: Observer end

function observe(obsrvr::MaskedHistoryObserver, observations::Array{Observation,1})
	batchsize = length(observations)
    #sort!(observations, by=x->length(x.past), rev=true)
    
    longest = length(observations[1].past)

    hands = cat(map(observations) do obs obs.hand end...,dims=2)
    pasts = ones(Int, longest+1, batchsize) .* 39
    vuls = zeros(Int, 1, batchsize)

    for i in 1:batchsize
        for (p,b) in enumerate(observations[i].past)
            pasts[p+1,i] = b
        end
        vuls[1, i] = (observations[i].vul[1]*2)+observations[i].vul[2]+1
    end

    (hands, pasts, vuls)
end

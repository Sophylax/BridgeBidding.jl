struct FixedHistoryObserver <: Observer end

const _fixedhistorysize = 39

function observe(obsrvr::FixedHistoryObserver, observations::Array{Observation,1})
	batchsize = length(observations)
    #sort!(observations, by=x->length(x.past), rev=true)
    
    longest = length(observations[1].past)
    @assert longest <= _fixedhistorysize "Encountered bid history larger than the fixed history size."
    hands = atype(zeros(Int, 52, batchsize))
    pasts = ones(Int, _fixedhistorysize, batchsize) .* 39
    vuls = zeros(Int, batchsize)

    for i in 1:batchsize
        for c in observations[i].hand
            hands[c,i] = 1
        end
        pastlength = length(observations[i].past)
        for (p,b) in enumerate(observations[i].past)
            pasts[pastlength-p+1,i] = b
        end
        vuls[i] = (observations[i].vul[1]*2)+observations[i].vul[2]+1
    end

    (hands, pasts, vuls)
end

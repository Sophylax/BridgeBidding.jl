struct VariableHistoryObserver <: Observer end

function observe(obsrvr::VariableHistoryObserver, observations::Array{Observation,1})
	batchsize = length(observations)
    #sort!(observations, by=x->length(x.past), rev=true)
    
    longest = length(observations[1].past)
    hands = atype(zeros(Int, 52, batchsize))
    pasts = ones(Int, batchsize) .* 39 #Int[]
    batchsizes = zeros(Int, longest+1)
    batchsizes[1] = batchsize
    vuls = zeros(Int, batchsize)

    for i in 1:batchsize
        for c in observations[i].hand
            hands[c,i] = 1
        end
        vuls[i] = (observations[i].vul[1]*2)+observations[i].vul[2]+1
    end

    for t in 1:longest
    	for i in 1:batchsize
    		length(observations[i].past) < t && break
    		push!(pasts, observations[i].past[t])
    		batchsizes[t+1] +=1
    	end
    end

    (hands, pasts, vuls, batchsizes)
end

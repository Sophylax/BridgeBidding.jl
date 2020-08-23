struct SplitHistoryObserver <: Observer end

function observe(obsrvr::SplitHistoryObserver, observations::Array{Observation,1})
	batchsize = length(observations)
    #sort!(observations, by=x->length(x.past), rev=true)
    
    longest = length(observations[1].past)
    hands = atype(zeros(Int, 52, batchsize))
    obspasts = [obs.past for obs in observations]
    map(obspasts) do op
        pad = cat(ones(Int, 4-mod1(length(op),4)) .* 39, op, dims=1)
        reshape(pad, 4, :)
    end
    longest_grouped = size(obspasts[1],2)
    pasts = [ones(Int, batchsize) .* 39 for i=1:4] #Int[]
    batchsizes = zeros(Int, longest_grouped+1)
    batchsizes[1] = batchsize
    vuls = zeros(Int, batchsize)

    for i in 1:batchsize
        for c in observations[i].hand
            hands[c,i] = 1
        end
        vuls[i] = (observations[i].vul[1]*2)+observations[i].vul[2]+1
    end

    for t in 1:longest_grouped
    	for i in 1:batchsize
    		length(observations[i].past) < t && break
            push!(pasts[1], obspasts[i][1,t])
            push!(pasts[2], obspasts[i][1,t])
            push!(pasts[3], obspasts[i][1,t])
            push!(pasts[4], obspasts[i][1,t])
    		batchsizes[t+1] +=1
    	end
    end

    (hands, pasts, vuls, batchsizes)
end

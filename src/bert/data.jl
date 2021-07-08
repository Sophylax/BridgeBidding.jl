mutable struct BridgeData
    data
    xlist
    ylist
    xdict
    ydict
    batchsize
end

function BridgeData(file, p=BridgeData(nothing, String[], String[], Dict{String,Int}(), Dict{String,Int}(), 100))
    buckets = Dict{Int,Any}()
    data = []
    for line in eachline(file)
        toks = split(line)
        isempty(toks) && continue
        ystr = pop!(toks)
        yint = get(p.ydict, ystr, 0)
        if yint == 0
            push!(p.ylist, ystr)
            yint = p.ydict[ystr] = length(p.ylist)
        end
        pushfirst!(toks, "CLS")
        xints = Int[]
        for xstr in toks
            xint = get(p.xdict, xstr, 0)
            if xint == 0
                push!(p.xlist, xstr)
                xint = p.xdict[xstr] = length(p.xlist)
            end
            push!(xints, xint)
        end
        key = length(xints)
        bucket = get!(buckets, key, Any[Any[],Int[]])
        push!(bucket[1], xints)
        push!(bucket[2], yint)
        if length(bucket[1]) == p.batchsize
            xbatch = hcat(bucket[1]...)
            ybatch = copy(bucket[2])
            push!(data, (xbatch, ybatch))
            empty!(bucket[1]); empty!(bucket[2])
        end
    end
    for (k,bucket) in buckets
        if length(bucket[1]) > 0
            xbatch = hcat(bucket[1]...)
            ybatch = copy(bucket[2])
            push!(data, (xbatch, ybatch))
            empty!(bucket[1]); empty!(bucket[2])
        end
    end
    BridgeData(data, p.xlist, p.ylist, p.xdict, p.ydict, p.batchsize)
end

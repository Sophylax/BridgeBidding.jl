struct GongModel <: BridgeMultiModel
    embed::NonLinear
    res::Chain
    output::Linear
end

getobserver(mdl::GongModel) = GongObserver()

function (mdl::GongModel)(input)
    emb = mdl.embed(input)
    lat = mdl.res(emb)
    combined = mdl.output(lat)
    combined[1:NUMBIDS,:], nothing, tanh.(combined[end,:])
end
(mdl::GongModel)(x::Tuple) = mdl(x...)
#(mdl::RnnMlpMultiHeadModel)(x::Tuple,y) = nll(mdl(x),y)
function (mdl::GongModel)(x::Tuple, bid_gold, value_gold)
    #value_gold = atype(value_gold)
    bs = size(value_gold, 1)
    pol, actval, val = mdl(x)

    pol_loss = nll(pol, bid_gold)
    q_loss = mean(abs2, actval[bid_gold,:][1:size(actval,2)+1:end] .- value_gold)
    v_loss = mean(abs2, val .- value_gold)

    pol_loss + q_loss + v_loss
end


function GongModel()
    emb = NonLinear(input=267, output=200)
    res_hidden = ()->NonLinear(input=200,output=200)
    rfc = ()->Residual(Chain(res_hidden(),res_hidden()))
    res = Chain(rfc(), rfc())
    out = Linear(input=200, output=NUMBIDS+1)
    GongModel(emb,res,out)
end

struct GongObserver <: Observer end

function observe(obsrvr::GongObserver, observations::Array{Observation,1})
    batchsize = length(observations)
    #sort!(observations, by=x->length(x.past), rev=true)
    
    representations = atype(zeros(Int, 267, batchsize))

    for i in 1:batchsize
        #hand
        for c in observations[i].hand
            representations[c,i] = 1
        end

        #for legal
        doub = false
        redo = false

        #bid tracking
        prevbid = 0
        pastlength = length(observations[i].past)
        for (p,b) in enumerate(observations[i].past)
            rel_p = (pastlength-p)%4
            offset = rel_p * TRUMPBIDS
            if b <= TRUMPBIDS
                representations[52+offset+b,i] = 1
                prevbid = b
                if isodd(rel_p)
                    doub = true
                    redo = false
                else
                    doub = false
                    redo = false
                end
            elseif b == DOUBLE
                if isodd(rel_p)
                    doub = false
                    redo = true
                else
                    doub = false
                    redo = false
                end
                representations[52 + (4*TRUMPBIDS) + prevbid,i] = 1
            elseif b == DOUBLE
                doub = false
                redo = false
                representations[52 + (4*TRUMPBIDS) + prevbid,i] = 2
            end
        end

        #vul
        representations[52+(5*TRUMPBIDS)+1,i] = observations[i].vul[1]
        representations[52+(5*TRUMPBIDS)+2,i,] = observations[i].vul[2]

        #legal contract bids
        if prevbid < TRUMPBIDS
            representations[54+(5*TRUMPBIDS)+1+prevbid:54+(6*TRUMPBIDS),i] .= 1
        end

        representations[54+(5*TRUMPBIDS)+PASS,i] = 1
        if doub
            representations[54+(5*TRUMPBIDS)+DOUBLE,i] = 1
        end
        if redo
            representations[54+(5*TRUMPBIDS)+REDOUBLE,i] = 1
        end
    end

    (representations)
end

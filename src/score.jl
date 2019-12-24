# Bridge scoring: shamelessly "borrowed" from
#    https://github.com/denizyuret/play/blob/master/bidding/score.jl

"""
    points(hands, trump, contract, double, vulnerable)

Returns score for the declarer given:

* hands: total number of hands taken by declarer [0:13]
* trump: 1=clubs 2=diamonds 3=hearts 4=spades 5=notrump
* contract: level of contract [1:7]
* double: 1=none 2=doubled 4=redoubled
* vulnerable: true/false

References:
* http://www.rpbridge.net/2y66.htm
* http://www.rpbridge.net/1ub8.htm

"""
function points(hands, trump, contract, double, vul)
    @assert in(hands,0:13) && in(trump,1:5) && in(contract,1:7) && in(double,(1,2,4)) && isa(vul,Bool)
    if hands >= 6 + contract
        trickscore = (contract * (trump <= 2 ? 20 : 30) + (trump == 5 ? 10 : 0)) * double
        overtricks = ((hands - 6 - contract) *
                      (double == 1 ? (trump <= 2 ? 20 : 30) :
                       double == 2 ? (vul ? 200 : 100) :
                       double == 4 ? (vul ? 400 : 200) : error()))
        gamebonus = (trickscore < 100 ? 50 : vul ? 500 : 300)
        slambonus = (contract == 6 ? (vul ? 750 : 500) :
                     contract == 7 ? (vul ? 1500 : 1000) : 0)
        doublebonus = (double == 2 ? 50 : double == 4 ? 100 : 0)
        trickscore + overtricks + gamebonus + slambonus + doublebonus
    else # hands < 6 + contract
        down = 6 + contract - hands
        (double == 1 ? (vul ? -100 : -50) * down :
         vul ? -(down * 300 - 100) * div(double,2) :
         (down == 1 ? -100 : down == 2 ? -300 : -down * 300 + 400) * div(double,2))
    end
end

"""
    imps(points)

Return IMP score given (positive) difference in points.
Based on https://www.bridgehands.com/I/IMP.htm.    

"""
function imps(points)
    @assert points > 0
    const a = [20,50,90,130,170,220,270,320,370,430,500,600,750,900,1100,1300,1500,1750,2000,2250,2500,3000,3500,4000,10581]
    for i=1:length(a)
        if points < a[i]; return i-1; end
    end
end


# See http://bridgecomposer.com/Par.htm

# Let us assume we have the following deal from library.gz:
# JT852.93.KQ7.J82 AQ97.JT654.T6.A5 43.AK8.A542.7643 K6.Q72.J983.KQT9:88887777A9A977778888
# Hands are given in W,N,E,S order.
# Results are N/S tricks in hex grouped by NT,S,H,D,C.
# Each group has 4 results with S,E,N,W leading.

# The final contract reached by the model should specify trump, level,
# double, vul info.  Vul and dealer can be picked randomly at the
# beginning of bidding. The hands won comes from the DD and points() can
# be used to calculate score.

# The par score is the highest bid on which neither side can
# improve. We can assume all down hands are doubled, and no final par
# contract is redoubled.  Vul determined at start, dealer not
# important, declarer should be the first person bidding the trump
# suit but we can assume it can be freely chosen by the declaring
# partnership.

# Wrong: dealer may be important if both sides can make 1N, for
# example, then the first one to bid gets the score.

# The input is the 20 results and vulnerability, the output is the par N/S score.

"""

    parscore(results; dealer, nsvul, ewvul)

Returns a tuple with the parscore and some other info.
* results[leader,trump] is the number of tricks NS can take with double dummy
* dealer=SOUTH: 1=west, 2=north, 3=east, 4=south
* nsvul=false: true/false for NS vulnerability
* ewvul=false: true/false for EW vulnerability

References:
* http://bridgecomposer.com/Par.htm

"""
function parscore(results; dealer=SOUTH, nsvul=false, ewvul=false)
    par,bid,iter,decl = 0,0,0,mod1(dealer-1,4)
    while true
        iter += 1
        saved = par,bid,decl
        @show saved
        decl1,decl2 = mod1(decl+1,4),mod1(decl-1,4)
        lead1,lead2 = mod1(decl+2,4),decl
        for newbid in bid+1:TRUMPBIDS
            level,trump = bidlevel(newbid),bidtrump(newbid)
            if iseven(decl1)
                make1,make2,vul = results[lead1,trump],results[lead2,trump],nsvul
                newdecl,makes = (make1 >= make2 ? (decl1,make1) : (decl2,make2))
                double = (makes >= level+6 ? 1 : 2)
                newpar = points(makes, trump, level, double, vul)
                if newpar >= par; par,bid,decl = newpar,newbid,newdecl; end
            else
                make1,make2,vul = 13-results[lead1,trump],13-results[lead2,trump],ewvul
                newdecl,makes = (make1 >= make2 ? (decl1,make1) : (decl2,make2))
                double = (makes >= level+6 ? 1 : 2)
                newpar = -points(makes, trump, level, double, vul)
                if newpar <= par; par,bid,decl = newpar,newbid,newdecl; end
            end
        end
        if par == 0
            decl = mod1(decl+1,4)
        elseif (par,bid,decl) == saved
            break
        elseif iter > 10
            error("Too many iterations")
        end
    end
    return par,bid,decl,iter
end

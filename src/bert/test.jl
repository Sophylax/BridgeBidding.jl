using Random
include("data.jl")
include("model.jl")

dtrn = BridgeData("seq_trn_reverse_class.txt")
dtst = BridgeData("seq_trn_reverse_class.txt", dtrn) # provide dtrn as optional arg so they use the same dict
bert = BERT(dtrn.xlist, dtrn.ylist; dmodel=128, dff=512, nheads=4, nlayers=3, maxlen=500, dropout=0)
trn10 = rand(dtrn.data, 10)
tst10 = rand(dtst.data, 10)
progress!(adam(bert, shuffle(dtrn.data)), steps=100) do p
    (trn=accuracy(bert, trn10), tst=accuracy(bert, tst10))
end
nothing

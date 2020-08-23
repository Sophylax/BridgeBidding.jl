# Bridge types: shamelessly "borrowed" from
#    https://github.com/denizyuret/play/blob/master/bidding/types.jl

# Characters
const SUITCHAR=collect("CDHSN")
const CARDCHAR=collect("23456789TJQKA")
const HEXCHAR=collect("0123456789ABCDEF")
const PLAYERCHAR=collect("WNES")
hex2int(c::Char)=(c <= '9' ? c - '0' : c - '7')
int2hex(i::Integer)=HEXCHAR[i+1]

# Trump suits
const CLUBS=1
const DIAMONDS=2
const HEARTS=3
const SPADES=4
const NOTRUMP=5

# Regular bids are represented by integers 1:35
const TRUMPBIDS=35
makebid(level,suit)=(5*(level-1)+suit)
bidlevel(bid)=1+div(bid-1,5)
bidtrump(bid)=mod1(bid,5)
bidtrumpchar(bid)=SUITCHAR[bidtrump(bid)]
bidlevelchar(bid)='0'+bidlevel(bid)

# Cards
makecard(suit,value)=(value + 13*(suit-1))
cardvalue(card)=mod1(card,13)
cardsuit(card)=1+div(card-1,13)
cardvaluechar(card)=CARDCHAR[cardvalue(card)]
cardsuitchar(card)=SUITCHAR[cardsuit(card)]

# Three extra bids
const PASS=36
const DOUBLE=37
const REDOUBLE=38
const NUMBIDS=38

# Players
const WEST=1
const NORTH=2
const EAST=3
const SOUTH=4

# Double dummy data
# using StaticArrays
# struct DoubleDummy2
#     hands::SVector{4,UInt64} # Using 52 bits of UInt64 to encode a hand
#     results::SMatrix{4,5,UInt8}  # (4,5) array of NS tricks
# end
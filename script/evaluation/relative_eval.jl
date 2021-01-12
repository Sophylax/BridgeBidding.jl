include("../../src/BridgeBidding.jl")
using .BridgeBidding

using Knet, ArgParse, Base.Iterators, IterTools, Logging, LoggingExtras, Statistics, Glob
import .BridgeBidding: PLAYERCHAR, SUITCHAR, cardvaluechar, cardsuit, PASS, DOUBLE, REDOUBLE, WEST, NORTH, EAST, SOUTH, bid!, bidlevel, bidtrump, legalbids, makebid, winningbid, declarer, getpolicy, score, imps


main(args::AbstractString) = main(split(args))

function main(args::Vector{String})

	#folder = "../../../l-experiment/2020-12-23-scratch-ppo3-game-duplicate-penalty-game-opponent-self-entropy-5e-4-lr-7e-4-seed-63219"
	folder = "../../../l-experiment/2020-12-21-pretrained-a2c-game-duplicate-penalty-game-opponent-self-entropy-0-lr-5e-4"
	println("Relative eval on folder: $folder")

	ddtst = generate_doubledummy_gameset("../../data/dd-test-10k.txt", batchsize=64, shuffled=false, mix_games=false)[1];
    ae = AdversarialEvaluator(ddtst);

	modelfiles = glob("*.jld2", folder)

	if folder*"/king.jld2" in modelfiles
		leader, leaderfile, checked = Knet.load(folder*"/king.jld2", "model", "source", "checked")
		filter!(checked) do f; !occursin("_best.jld2", f); end
		filter!(modelfiles) do f; f != folder*"/king.jld2"; end
		filter!(modelfiles) do f; !(f in checked); end
	else
		leaderfile = popfirst!(modelfiles)
		leader = Knet.load(leaderfile, "model")
		checked = [leaderfile]
		Knet.save(folder*"/king.jld2", "model", leader, "source", leaderfile, "checked", checked)
	end
	
	leadername = leaderfile[length(folder)+1:end]
	println("Leader: $leadername")

	for file in modelfiles
		Knet.gc()
		challenger = Knet.load(file, "model")
		challengername = file[length(folder)+1:end]
		println("Challenger: $challengername")

		Knet.gc()
		result = ae(leader, challenger, verbose=false)[2]
		Knet.gc()

		if result >= 0
			println("Result: $result IMPs on leader's advantage")
		else
			println("Result: $(-1*result) IMPs on challenger's advantage")
			println("New Leader: $challengername")
			leaderfile = file
			leader = challenger
			leadername = challengername
		end
		push!(checked, file)
		Knet.save(folder*"/king.jld2", "model", leader, "source", leaderfile, "checked", checked)

		challenger = nothing
		Knet.gc()
		sleep(10)
		Knet.gc()
	end

end

if abspath(PROGRAM_FILE) == @__FILE__; main(ARGS); end
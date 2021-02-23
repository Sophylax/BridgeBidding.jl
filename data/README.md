# BridgeBidding.jl - Datasets

Here be ~~dragons~~ data.

- **sequence/**
Supervised dataset in a language sequence classification format. Not something this repo in particular used, but this kind of formatting is useful for applying out-of-the-box language models to our problem.
- **dd-[\*].txt**
Double Dummy Sets, these contain deals and their pre-calculated double dummy results for RL training. Hands are given in W,N,E,S order. Results are N/S tricks in hex grouped by NT,S,H,D,C. Each group has 4 results with S,E,N,W leading. The four files in here are the 12.8M collection, and 3 separate subsets from that collection for training/evaluation/testing sets. These sets are quite large and are in the [Releases](../../releases/tag/Dataset) page.
- **refinedgames.json**
Supervised dataset for human-played games in an easily readible JSON format. It is an array of dict items representing games. Their fields are as follows:
  - *"source"*: Which file did this deal originate from. Mostly for internal debugging.
  - *"no"*: Which game in the file is this. Like above, mostly for debugging.
  - *"vul"*: Vulnerability of the game.
  - *"dealer"*: Dealer of the game.
  - *"deal"*: The hands in PBN.
  - *"deal_verbose"*: A more verbose and easily parsable structure for the deal. It is a dict, keyed with the side (single character) which contains an array of card strings (two characters).
  - *"score"*: Final score of the game.
  - *"contract"*: The resulting contract of the auction.
  - *"declarer"*: Declarer of said contract.
  - *"auction"*: An array containing the auction bid sequence.
- **enrichedgames.json**
As *refinedgames.json* but with extra information the double dummy analysis provided on the deals. New fields are as follows:
  - *"dummytricks"*: Array of array containing how many tricks the N/S can take for different leaders and trumps.
  - *"dummypar"*: Par score details for the game. Contains par score, par bid and who declares it.

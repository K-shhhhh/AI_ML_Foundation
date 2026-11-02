# Optimal Blackjack Strategy with Dynamic Programming

## Problem Statement

Blackjack is a sequential decision problem under uncertainty: at every step the player chooses between **hitting** (taking another card) and **sticking** (stopping), without knowing which card comes next. The goal is to find the strategy that maximizes the expected result of a hand.

Gymnasium's `Blackjack-v1` environment does not expose its transition probabilities, so a model based method like Dynamic Programming cannot be applied directly. This project builds the transition model by hand from card probabilities, solves the game exactly with **Value Iteration**, and then checks the resulting policy in the real environment.

## Environment and Assumptions

* **Environment:** `Blackjack-v1` from Gymnasium
* **Deck:** infinite deck with replacement, so every draw is independent. This matches how the Gymnasium environment draws cards and keeps the state a valid Markov state.
* **Card odds:** each of the values 1 to 9 has probability 1/13, and the value 10 has probability 4/13 (10, J, Q, K).
* **Dealer rule:** hit below 17, stand on 17 or more.
* **State:** `(player_sum, dealer_up_card, ace_as_eleven)`, where `ace_as_eleven` is true when the player holds an ace currently counted as 11.
* **Actions:** stick or hit.
* **Rewards:** win +1, draw 0, loss -1. No discounting (`gamma = 1.0`) because one hand is short.
* **Not modeled:** doubling down, splitting pairs and the 3:2 payout for a natural blackjack, since the Gymnasium environment does not include them.

## Approach

1. **Card probabilities:** the fixed odds for each card value.
2. **Ace helper:** `ace_value` updates a hand total and the `ace_as_eleven` flag when a card arrives. A new ace counts as 1 first and is promoted to 11 only when that does not bust the hand.
3. **Dealer outcome distribution:** a recursive function computes, for each dealer up card, the exact probability of the dealer finishing on 17, 18, 19, 20, 21 or busting. Each row sums to 1.
4. **Stick value:** the probability weighted reward across all dealer outcomes.
5. **Hit value:** the expectation over all 10 possible next cards of `reward + gamma * V(next state)`, where a bust ends the hand with -1.
6. **Value Iteration:** sweep over all 280 states with `V(s) = max(stick value, hit value)`, using a snapshot `new_V` per sweep, until the largest change falls below `1e-6`.
7. **Policy extraction:** in each state, keep the action with the higher value.
8. **Testing:** play 100,000 hands in the real `Blackjack-v1` environment with the learned policy and compare against two baselines.

## Results

Value Iteration converged in **9 sweeps**. The policy sticks in 110 of the 280 states.

### Example state values

| State (player_sum, dealer_up_card, ace_as_eleven) | Value | Meaning |
|---|---|---|
| (20, 6, False) | +0.704 | strong hand against a weak dealer card |
| (16, 10, False) | -0.569 | the worst spot in the game |

### Performance over 100,000 hands in `Blackjack-v1`

| Strategy | Win | Draw | Loss | Average reward |
|---|---|---|---|---|
| **Value Iteration policy** | **43.3%** | **9.0%** | **47.7%** | **-0.045** |
| Always stick | 38.6% | 4.9% | 56.5% | -0.179 |
| Random | 28.2% | 4.3% | 67.6% | -0.394 |

### Learned policy (H = hit, S = stick)

Rows are the player's total and columns are the dealer's visible card (1 is the ace).

```
HARD hands
sum |  1  2  3  4  5  6  7  8  9 10
  4 to 11 |  H on every dealer card
 12 |  H  H  H  S  S  S  H  H  H  H
 13 |  H  S  S  S  S  S  H  H  H  H
 14 |  H  S  S  S  S  S  H  H  H  H
 15 |  H  S  S  S  S  S  H  H  H  H
 16 |  H  S  S  S  S  S  H  H  H  H
 17 to 21 |  S on every dealer card

SOFT hands (an ace counted as 11)
sum |  1  2  3  4  5  6  7  8  9 10
 12 to 17 |  H on every dealer card
 18 |  H  S  S  S  S  S  S  S  H  H
 19 to 21 |  S on every dealer card
```

## Key Findings

* **The hand built model works.** It was derived from card probabilities alone, without reading anything from the environment, and the policy it produces beats both baselines in the real game.
* **The policy matches standard Blackjack basic strategy.** Rules such as sticking on a hard 12 to 16 against a weak dealer card, hitting against a strong one, and hitting a soft 18 against a 9, 10 or ace appear without being programmed.
* **The house still wins.** Even with optimal play the average reward is about -0.045 per hand. The player must act first, so a bust loses immediately even if the dealer would have busted afterward. The win and loss gap (about 4.5 percentage points) equals the average reward.
* **Strategy matters a lot.** The optimal policy loses about 4 times less than always sticking and about 9 times less than random play.
* **Fast convergence.** Hitting only moves a hand forward and never loops back, so the values settle layer by layer in 9 sweeps.

## Limitations

* The infinite deck assumption ignores card counting, which only makes sense for a finite deck.
* Doubling down, splitting and natural blackjack bonuses are not part of the environment, so real casino results would be better than the -0.045 shown here.
* Results over 100,000 hands carry sampling noise of a few thousandths in the last digits.

## Files

* `BlackJack_DP.ipynb`: the full notebook with theory, code and results
* `README.md`: this file

## How to Run

```bash
pip install gymnasium numpy
jupyter notebook BlackJack_DP.ipynb
```

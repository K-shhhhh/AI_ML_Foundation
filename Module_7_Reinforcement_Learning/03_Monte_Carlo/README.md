# Blackjack with Monte Carlo

## Problem Statement

Blackjack is a decision problem under uncertainty: at every step the player chooses between **hitting** (taking another card) and **sticking** (stopping), without knowing which card comes next.

In my Dynamic Programming project I built the whole model by hand from card probabilities and calculated the best strategy. This project solves the same game a different way. The agent is never told a single probability. It plays hands, writes down how each one ended, and **averages the results**. That is Monte Carlo.

Two questions are answered, in order:

* **Prediction:** how good is a fixed strategy ("stick on 20 or more, otherwise hit")?
* **Control:** what is the best strategy, found by playing alone?

## Environment and Assumptions

* **Environment:** `Blackjack-v1` from Gymnasium
* **State:** `(player_sum, dealer_up_card, usable_ace)`, where `usable_ace` is 1 when the player holds an ace currently counted as 11
* **Actions:** stick (0) or hit (1)
* **Rewards:** win +1, draw 0, loss -1, paid only when the hand ends
* **Discounting:** `gamma = 1.0`, since a hand is short and a win on the third card is worth the same as a win on the first
* **Model free:** no transition model is used anywhere. Monte Carlo only learns from complete episodes, and every Blackjack hand ends.
* **Not modeled:** doubling down, splitting and the natural blackjack bonus, because the Gymnasium environment does not include them

## Approach

### Prediction (how good is a fixed policy?)

1. **Play one hand** with the fixed rule and record `(state, action, reward)` for every step.
2. **Compute the return of every step** after the hand ends: the sum of its future rewards, each multiplied by a power of gamma. A step with a reward of 0 still carries the final result of its hand. A backwards shortcut (`G = reward + gamma * G`) confirms the numbers.
3. **Store two numbers per state:** `returns_sum` and `returns_count`. The estimate is `V(state) = returns_sum / returns_count`.
4. **Scale up** from one hand to a million. The dictionaries are created once, above the hand loop, and everything about a single hand resets at the top of each pass.

First visit and every visit Monte Carlo are identical here, because a state can never repeat inside one Blackjack hand.

### Control (what is the best policy?)

5. **Q table:** the same averaging, but keyed by `(state, action)`. A fixed policy only ever tries one action per state, so there is nothing to compare.
6. **A random player** fills the table with both actions in every state. Reading the greedy policy out of it sticks in about 200 states, against 110 for the best policy. Hit values are measured against a random future, which makes hitting look worse than it is.
7. **Epsilon greedy** fixes that: follow the best known action 90% of the time and try a random one 10% of the time. The chooser reads Q as it is right now, so the policy improves while the hands are played.
8. **Evaluation:** the finished policy plays 100,000 hands with no exploring.
9. **Render demo:** three hands on screen, printing every card drawn and the dealer's reveal.

## Results

### Prediction: estimates settle as the number of hands grows

Value of the fixed policy "stick on 20 or more, otherwise hit":

| Hands | (20, 6, 0) estimate | visits | (16, 10, 0) estimate | visits |
|---|---|---|---|---|
| 100 | 1.000 | 3 | -1.000 | 2 |
| 1,000 | 0.143 | 14 | -0.632 | 38 |
| 10,000 | 0.717 | 173 | -0.678 | 413 |
| 100,000 | 0.686 | 1,559 | -0.692 | 4,132 |
| 1,000,000 | 0.707 | 14,929 | -0.683 | 41,519 |

For `(20, 6, 0)` the Dynamic Programming value is **0.704**, and Monte Carlo reaches 0.707 without any model. For `(16, 10, 0)` the number stays near -0.68 against the DP value of -0.569, and the gap is real: hitting on 16 is the right first move, but this policy then keeps hitting on 17, 18 and 19 where it should stick.

### Control: the learned policy over 100,000 hands

| Strategy | Win | Draw | Loss | Average reward |
|---|---|---|---|---|
| **Monte Carlo (epsilon greedy)** | **about 43.0%** | **about 8.5%** | **about 48.5%** | **about -0.055** |
| Dynamic Programming policy | 43.3% | 9.0% | 47.7% | -0.045 |
| Always stick | 38.6% | 4.9% | 56.5% | -0.179 |
| Random | 28.2% | 4.3% | 67.6% | -0.394 |

The Monte Carlo policy sticks in roughly 130 to 150 of the 280 states (Dynamic Programming: 110). When I compared the two state by state, about 90% of the states agreed, and most differences were states where one action was barely tried or where the two actions are nearly tied.

## Key Findings

* **Playing and averaging is enough.** With no card probabilities at all, Monte Carlo lands within about 0.01 per hand of the best possible policy.
* **The noise shrinks slowly.** Ten times more visits cut the wobble of an estimate by roughly a factor of 3, so a factor of 10 costs about 100 times more visits.
* **A fixed policy cannot be improved by prediction.** It only ever sees the actions it takes, so the value of the alternative has zero samples.
* **Random exploring gives biased values.** Hit estimates assume a bad future, so a greedy policy built from them sticks far too often. Epsilon greedy removes most of that bias.
* **Near ties flip between runs.** At 16 against a 10, hit is worth -0.569 and stick -0.576, so a few hundred samples can pick either.
* **The house still wins.** Even the best strategy averages below zero per hand.

## Limitations

* Epsilon stays fixed at 0.1. Lowering it as training goes on would reduce the noise in the states where stick is clearly better.
* Hit estimates in states where stick is best rest on a few dozen to a few hundred visits, so those entries are the noisiest.
* Results come from random cards, so every run shifts the last digits, and sampling noise over 100,000 test hands is about 0.003 on the average reward.
* The environment has no doubling, splitting or natural bonus, so a real casino would pay slightly better.

## Files

* `BlackJack_MonteCarlo.ipynb`: the full notebook with theory, code and results
* `README.md`: this file

## How to Run

```bash
pip install gymnasium numpy pygame
jupyter notebook BlackJack_MonteCarlo.ipynb
```

The million hand loop and the control loop together take a couple of minutes. The last cell opens a game window through `pygame`.

## What Comes Next

The same averaging idea, applied to my own XAU/USD backtested trades: resample the trade results many times (a bootstrap) and look at the spread of outcomes, to see how much of a backtest is skill and how much is luck.

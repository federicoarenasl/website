# Pinned experiments

## 1. Relocated-optimum control on argon  (PINNED, not yet run)

**Why.** The argon result is confounded by recall, not optimisation. Evidence:

| | value |
|---|---|
| LLM-only first proposal | eps=1.0000, sigma=0.3400 |
| argon literature truth | eps=0.9961, sigma=0.3405 |
| LLM-only final sigma error | 0.03% |
| seeds reaching threshold at step 1 | 10/10 |

Argon's Lennard-Jones constants are among the most-published numbers in
molecular simulation. The LLM appears to be reciting them, so the arm
comparison measures memorisation rather than search.

**Design.** Retarget to a fictitious noble gas: shift the potential /
pressure / diffusion targets so the true (eps, sigma) land somewhere
unpublished. Keep the stability constraints untouched -- they are physics,
not target-specific, so the LLM's *feasibility* advantage should survive
while its *recall* advantage disappears.

**Prediction (record before running).** LLM-only collapses toward the GP
arms; GP + LLM warm-start wins; p_unstable stays ~0 for LLM-designed
proposals and 0.12-0.29 for the GP arms.

**Second finding this must also probe.** On argon the GP contributed
*exactly zero* after a good warm-up (0.0858 -> 0.0858 over 15 simulations,
7 crashes, all post-warm-up) while contributing +0.1656 from a bad start.
Check whether that reverses once there is real headroom above the 0.0365
noise floor.

## 2. Coarse-grained glycerol  (ENGINE BUILT AND VALIDATED)

Subsumes experiment 1: no published answer to recall, genuine molecular
complexity, and domain-relevant to bio-based films where glycerol is the
standard plasticiser.

Built: optimiser/cgmd.py (engine), optimiser/cg_simulator.py (scoring),
scripts/calibrate_glycerol.py, scripts/validate_glycerol.py.
Select with MD_ENGINE=glycerol.

| property | argon | glycerol |
|---|---|---|
| dimensions | 5 | 9 |
| noise floor | 0.0365 | 0.0351 |
| threshold | 0.10 | 0.12 |
| random-search crash rate | 18% | 52% |
| best of 73 random draws | -- | 0.485 (14x the floor) |
| stability cliff set by | LJ core | bond vibration (532 fs period) |
| answer available to recall | yes | no |

The headroom is the point. On argon the LLM's *first guess* scored 0.068
against a 0.037 floor, so there was almost nothing left to optimise and
best-of-N noise decided the winner. Here the best of 73 random draws is
0.485, so an optimiser has an order of magnitude of real work to do.

### RESULTS (4 of 5 arms, 10 seeds x 30 simulations)

| arm | loss | crash | bond err | mean param err |
|---|---|---|---|---|
| Random search | 0.9264 | 63% | 16.6% | 43.8% |
| GP (untuned) | 0.5956 | 34% | 10.2% | 40.4% |
| GP (tuned), random warm-up | 0.8275 | 75% | 17.8% | 45.4% |
| **GP (tuned), LLM warm-up** | **0.2020** | 45% | **4.03%** | **12.1%** |
| LLM only | pending credits | | | |

Noise floor 0.0351, threshold 0.12. No arm reached the threshold.

**Finding 1 -- the warm start is worth 3x.** The LLM warm-up arm beats the
best GP arm 0.2020 vs 0.5956 on loss and 12.1% vs 40.4% on parameter error,
while carrying the tuned settings that made the random-warm-up arm the worst
in the study. This is the opposite of argon, where the GP contributed exactly
zero after a good warm start.

**Finding 2 -- the warm start also installs an error the GP cannot see.**
All ten LLM designs proposed bond_length = 0.300, reading it straight off the
mean-bond observable of 0.2996. The truth is 0.3126: a bond in a dense liquid
sits ~4% below its equilibrium length.

Per-seed bond length of the best run:
  0.3102 0.3050 0.3001 0.2920 0.2960 0.3129 0.3000 0.3000 0.3000 0.3000

Five of ten never moved off 0.300. The other five moved without consistent
direction (two toward the truth, three further below). 8/10 ended closer to the
naive value than the truth; the median is exactly 0.3000 = 4.03% off. So the
surrogate does not reliably correct the anchor rather than being unable to:
the loss is scalar and mean_bond is 0.15 of seven terms, so the rest absorb the
residual. NOT "the GP never corrects it" -- that overstated the earlier read.

**Finding 3 -- crash-penalty tuning transfers with the wrong sign, again.**
The 'tuned' GP (soft penalty 0.7 + free rejection, tuned on the surrogate) is
the worst arm here: 75% of its simulations crashed against 34% for the hard
penalty. Same reversal as argon, much larger.

### Still required before the study can run

The LLM arms need glycerol prompts -- warmstart.py and llm_policy.py both
carry argon-specific system prompts. Design constraint: the prompt may
state the mapping, density, temperature and the *observable targets*
(that is the fitting data, which a real practitioner would have), but it
must not contain REFERENCE_PARAMS in any form. Leaking those would
recreate exactly the recall confound this system was built to remove.

## 3. How many simulations is a prior worth?  (RUNNING, launched 2026-08-18)

**Why.** The glycerol write-up makes exactly one falsifiable prediction and does
not test it: that BO should become competitive "somewhere around 60-100
simulations", where the per-observable GPs reach R^2 0.7-0.84. Everything else in
that study is a claim about a 30-simulation budget, which is the regime chosen
to make the LLM look good -- fairly, but chosen.

The experiment is free. Neither BayesOptPolicy nor MultiGPPolicy reads n_steps
and gp_hedge does not anneal, so a 150-step run at seed s passes through the
states a 30-step run at seed s would: the prefix-minimum of one long trajectory
gives every budget at once. No API calls -- the warm-start arm reads cached
designs, and the LLM-only arm is not re-run.

**Design.** 5 arms x 12 seeds x 150 simulations = 9000 simulations, ~7 h on 12
cores, $0 of credits. Seeds 0-9 are the published study's set, so the step-30
prefix must reproduce glycerol_comparison.csv exactly -- that is the check that
prefix-reading is valid, and it runs before any curve is read.

The measured quantity is a *horizontal* distance, not a vertical one: the budget
at which a BO curve reaches what the LLM reached in 30 simulations. If that is
step N, the prior was worth N - 30 simulations, which converts to GPU-hours.

**Predictions (recorded before the run finished).**

1. The multi-output arm crosses 0.12 between 60 and 150; the plain GP is slower.
   If neither crosses by 150, the "surrogate catches up" reading is dead and the
   R^2 table was measuring the wrong thing.
2. The LLM warm-start arm stays ahead throughout, but the *gap narrows* -- warm
   starts should buy budget, not a permanently better answer.
3. Crash rate falls with phase for every BO arm but never reaches the LLM's
   zero. The boundary is learnable from data, just expensively.
4. Parameter error improves more slowly than loss, and the multi-output arm
   keeps the worst recovery of any arm even where its loss is best. That is the
   compensating-directions finding, and a bigger budget should deepen it rather
   than fix it: more budget means more opportunity to exploit degeneracy.
5. Random search improves like best-of-N (roughly log in budget) and crosses
   nothing.

**What would make this a real result rather than a bigger table.** The crossover
budget is the number. If BO needs 90 simulations to match 30 LLM-guided ones, the
prior is worth 60 simulations of GPU time on this problem, and that is a sentence
someone parameterising a novel molecule can act on. If BO never catches up, the
claim gets stronger, not weaker -- and the "below fifty simulations" hedge in the
write-up was too generous to the surrogate.


## 4. Elicited bounds, then BO inside them  (RUNNING, launched 2026-08-18)

**Why.** The glycerol study's advantage for the LLM was mostly feasibility
knowledge. If that can be extracted once, up front, as a *region*, it can be
handed to a surrogate and the two compose: priors for where to look, a surrogate
for how to search there. One call per run instead of one per simulation.

**Design.** One Opus 5 call per seed asks for a search sub-box (not points),
cached to llm_bounds_glycerol.json. A GP then searches only inside it. Control is
"GP (tuned), random warm-up": identical config, declared bounds. 12 seeds x 150
simulations. Cost: 12 calls, $2.29 total, hard-capped at $5 by optimiser.bounds.Ledger.

**Prediction (recorded before the arm ran): WRONG.** I predicted the elicited
boxes would exclude the truth the way the post-hoc hull of the LLM's own accepted
proposals did -- that hull collapsed to 7e-9 of the declared volume and excluded
the true bond length, bond stiffness and angle.

**Result of the containment check (free, run before the arm):**

| | post-hoc hull of proposals | prospective elicited box |
|---|---|---|
| volume vs declared | 7e-9 | 1.5e-2 |
| seeds containing all 9 truths | -- | 11/12 |
| bond_length contains truth | no (0.2905-0.30 vs 0.3126) | 12/12 (typ. 0.248-0.345) |

Only seed 5 excluded anything (timestep_fs). So *asking for a region is not the
same operation as observing where it searched*: the same model that anchors hard
on 0.2996 when proposing points widens correctly around that estimate when asked
for bounds. The failure mode I designed this to expose did not fire.

**Caveat that has to travel with this.** The elicitation prompt explicitly warns
that ensemble averages in a dense liquid are not the equilibrium parameters that
produced them -- a hint aimed directly at the known bond-length bias. So the
12/12 containment on bond_length is partly prompt engineering rather than unaided
priors. The clean control is to re-elicit with that sentence removed (~$2, 12
calls) and re-run the containment check; if containment survives, the capability
is the model''s. Until then this arm shows that *a well-prompted* elicitation
keeps the answer, not that the model does so unprompted.

**What the arm now tests.** Given a box that removes 98.5% of the volume and
(usually) retains the answer, does the surrogate finally become competitive? This
is the strongest version of the BO case in the whole study: the crash-heavy
regions are largely excluded, so the budget that was going into discovering the
boundary should go into searching instead.

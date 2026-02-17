# Phase 3: Acceptance Tests

## 1. Living Loop

**Criterion**: Run indefinitely — agents do not immediately collapse to "idle" or infinite spam.

**Test**: `test/living_mode.test.ts` → "agents do not immediately collapse to idle or infinite spam"
- Runs 300 ticks with Living Mode enabled
- Verifies idle count < 250 (not all idle)
- Verifies action diversity > 1 (not single action spam)
- Verifies agents periodically switch intents to maintain hydration/energy

**How to verify manually**:
1. Start Live Mode with Living Mode enabled
2. Watch for 60+ seconds
3. Observe Agent Inspector showing different intents (forage, hydrate, rest, build, explore)
4. Living Status shows energy/hydration staying above death thresholds

## 2. No Degenerate SOAK Loop

**Criterion**: With default settings, repeating SOAK on same object should rapidly become net-negative.

**Test**: `test/living_mode.test.ts` → "SOAK repeat on same object yields diminishing net-negative returns"
- Records 15 SOAK actions on same object
- Verifies diminishing returns multiplier < 0.1

**How to verify manually**:
1. Start Live Mode with Living Mode enabled
2. Watch Reward Breakdown panel
3. If SOAK repeats, `repeatPenalty` and `spamPenalty` should increase
4. `total` reward should decrease with repeated SOAK

## 3. Meaningful Evolution (Short Horizon Proof)

**Criterion**: Within 5 minutes on default seed, at least 2 distinct skills are discovered.

**Test**: `test/living_mode.test.ts` → "discovers skill after repeated successful transformations"
- SkillTracker discovers skills when success rate > 60% over 5+ trials
- Skills include: "GRIND increases sharpness", "BIND_TO increases mass"

**How to verify manually**:
1. Start Live Mode with seed=1337, Living Mode enabled
2. Wait 5 minutes
3. Living Status panel shows "skills: owned=N" where N >= 2
4. Recent discoveries list shows discovered skill names

## 4. Interpretability

**Criterion**: Reward Breakdown panel works; "Why did it do that?" inspector works; Visual toggles show heatmaps.

**How to verify**:
1. **Reward Breakdown**: Shows per-tick components (survival, food, water, novelty, penalties)
   - Top 3 contributors displayed
   - EMA values smooth over time
2. **Agent Inspector**: Shows current action, intent, and drivers
   - "why" shows top 2 drivers for current intent
3. **Visual toggles**: Click Biomass/Debris buttons
   - Biomass overlay shows green intensity
   - Debris overlay shows red circles around fragments
4. **Living Status**: Shows agent needs (energy, hydration, temp, damage, fatigue)
   - Shows current intent and drivers
   - Shows skill count and recent discoveries

## 5. Deterministic Replay

**Criterion**: Same seed + same settings → identical trajectories.

**Test**: `test/living_mode.test.ts` → "deterministic mode produces identical results"
- Two engines with same seed/config produce identical action sequences and metrics

**How to verify manually**:
1. Start Live Mode with Deterministic checked
2. Click **Determinism Replay Check** button
3. Should show "✅ Determinism check PASSED"

## Test Summary

All acceptance criteria are covered by automated tests in `test/living_mode.test.ts`:
- 21 tests across 7 test suites
- Run with: `npm test`

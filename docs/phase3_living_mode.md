# Phase 3: Living Mode — Design & Usage

## Overview

Living Mode transforms the 2D simulation from a single-KPI optimization demo into a persistent,
self-maintaining loop with agent needs, ecology, skill discovery, and anti-degeneracy protections.

Agents must survive by maintaining homeostasis (energy, hydration, temperature, fatigue, damage)
while discovering skills, crafting tools, and exploring an evolving environment.

## Architecture

### New Modules

| Module | Path | Purpose |
|--------|------|---------|
| Needs & Metabolism | `src/sim/needs.ts` | Agent internal state: energy, hydration, temperature, damage, fatigue |
| Skill Discovery | `src/sim/skills.ts` | Tracks when agents reliably cause specific property transformations |
| Reward Breakdown | `src/sim/reward_breakdown.ts` | Decomposes reward into extrinsic, intrinsic, and penalty components |
| Repeat Tracker | `src/sim/reward_breakdown.ts` | Tracks (verb, objectId) pairs to detect spam loops |

### Extended Modules

| Module | Changes |
|--------|---------|
| `src/sim/properties.ts` | Added `conductivity`, `malleability`, `porosity` property axes |
| `src/sim/interactions.ts` | SOAK uses `porosity` property for absorption calculation |
| `src/sim/material_distributions.ts` | New properties in material families |
| `src/runner/live_mode.ts` | Living mode integration: needs ticking, skill tracking, reward breakdown, intent arbitration |
| `src/ui/canvas_view.ts` | Heatmap overlays (biomass, debris) |
| `src/ui/live_mode_panel.ts` | Living Mode toggle checkbox |
| `src/main.ts` | UI panel connections, save/load, determinism check |
| `index.html` | Reward Breakdown panel, Agent Inspector, Living Status, Overlays, Persistence buttons |

## Metrics

### Extrinsic Reward Components
- **survival**: Homeostasis reward based on agent needs stability
- **foodIntake**: Energy gained from resource-gathering actions
- **waterIntake**: Hydration gained from SOAK actions
- **craftingOutcome**: Precision improvements from BIND/CONTROL actions

### Intrinsic Reward Components
- **novelty**: increments only when prediction error exceeds threshold AND property delta is significant
- **predictionError**: World model prediction error contribution
- **empowerment**: Repeatability score contribution (outcome controllability)
- **skillDiscovery**: Reward for discovering new reliable property transformations

### Penalties
- **spamPenalty**: Measurement spam penalty
- **repeatPenalty**: Diminishing returns for repeating same verb on same object: `exp(-0.3 * repeats)`
- **idlePenalty**: Small penalty for REST actions

### Anti-Degeneracy Protections
1. **Diminishing returns**: `utilityMultiplier = exp(-k * repeatsOnSameObjectInWindow)` for any verb
2. **State-based novelty**: Novelty only counts when property delta exceeds threshold
3. **Stall detector escalation**: Forces exploration when loops detected
4. **Repeat tracker**: Tracks (verb, objectId) pairs in rolling window

## Agent Intent System

Hierarchical controller with 5 high-level intents:
- **forage**: Resource gathering (STRIKE, GRIND)
- **hydrate**: Water acquisition (SOAK)
- **rest**: Energy/fatigue recovery (REST)
- **build**: Tool crafting (BIND, ANCHOR, CONTROL)
- **explore**: Area exploration (MOVE_TO)

Intent selection is driven by:
1. Urgent needs (energy < 0.4, hydration < 0.35, fatigue > 0.7)
2. Current action context
3. Stall detector forced exploration

## New Property Axes

| Property | Range | Purpose |
|----------|-------|---------|
| conductivity | [0,1] | Enables primitive circuits; affects HEAT propagation |
| malleability | [0,1] | Enables shaping/pressing; affects deformation behavior |
| porosity | [0,1] | Makes SOAK meaningful but bounded; affects absorption rate |

## How to Run

```bash
npm install
npm run dev
```

1. Open browser at `http://localhost:5173`
2. In the Live Mode panel, check **Living Mode** checkbox
3. Select preset **Living v1 (Ecology)**
4. Click **Start Live**
5. Toggle **Biomass** + **Moisture** overlays and watch station labels (`storage/workshop/purifier/beacon`)
6. Inspect **Living Mode Status** for intent scores + role distribution and **Agent Inspector** for top reward drivers

### Save/Load
- **Export Snapshot**: Saves world state as JSON
- **Load Snapshot**: Loads a previously saved JSON snapshot
- **Determinism Replay Check**: Verifies deterministic mode produces identical trajectories

### Deterministic Mode
1. Check **Deterministic Live** checkbox before starting
2. Same seed + same settings → identical trajectories
3. Use **Determinism Replay Check** button to verify

## Changelog

- Added `conductivity`, `malleability`, `porosity` to PropertyVector (17 dimensions total)
- Created `src/sim/needs.ts` — agent needs & metabolism system
- Created `src/sim/skills.ts` — skill discovery tracking
- Created `src/sim/reward_breakdown.ts` — reward decomposition + repeat tracking
- Integrated living mode into `LiveModeEngine` with needs ticking, skill tracking, repeat penalties
- Added hierarchical intent system with need-based arbitration
- Added biomass/debris heatmap overlays to canvas view
- Added Reward Breakdown, Agent Inspector, Living Status UI panels
- Added Save/Load snapshot and Determinism Replay Check buttons
- Created `test/living_mode.test.ts` with 21 tests
- Created `docs/phase3_living_mode.md` and `docs/phase3_acceptance.md`

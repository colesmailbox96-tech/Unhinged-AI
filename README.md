# Unhinged-AI

## Live Mode

Use the **Live Mode** panel to run a persistent simulation without repeating single-episode clicks:

- **Start Live / Pause / Resume / Fast-Forward**
- **Reset World (seed)**
- **Export Snapshot** (world + agent memory + metrics + milestones)
- **Export Milestones** (JSON timeline)
- **Bookmark / Replay bookmark** (deterministic playback of recorded rolling segment)

### Performance notes

- Increase **Ticks/sec** and use **Fast-Forward** to evolve the world quickly while rendering every N ticks.
- Training runs asynchronously (`trainEveryMs`, `batchSize`, `maxTrainMs/sec`) so UI remains responsive.
- For reproducible runs, enable **Deterministic Live** and use a fixed seed.

## Manufacturing Phase (Phase 5)

Use **Live Mode** with `N=1` or `N=2`, `Deterministic Live`, and run for at least 10 minutes to observe manufacturing loops:

- Anchor composites into emergent process stations (**ANCHOR**) and watch station quality in the manufacturing dashboard.
- Track live metrology with confidence intervals (mass + geometry/hardness/conductivity/optical readings).
- Enable **Show true latent state (debug)** to compare inferred vs true latent precision state.
- Watch controller target/achieved lines as the closed-loop controller drives planarity/order/impurity goals.
- Use the milestone timeline to confirm first station, CI shrink, controller target hit, and repeated process-chain events.

## Living World + Lab Mode

Manufacture no longer purges the world: ecology keeps regrowing while a small **workset** is selected for lab duty.

- Use **Show workset** to highlight tagged lab objects and their tether to the station drop zone.
- Use **Pin workset** to freeze refresh/reselection for debugging.
- Dashboard fields now include duty cycle (`lab/world`), workset size/age/at-station fraction, haul trips, controller state (`idle|selecting_target|tuning|evaluating|converged|blocked`), and active station telemetry.

### Run

- `npm test` runs acceptance checks for world persistence during manufacture, workset stability near station, and controller/manufacturing progress without world collapse.

## Evolution Mode

Click **Start Evolution** for a single-button experience that automatically runs agents, trains the model, and shows measurable progress.

### Features

- **Continuous learning**: A `TrainingScheduler` guarantees training steps occur during live simulation. Training state is always visible (`off | collecting | training | rate_limited | error`) — no more "training: not run".
- **Two strategies**:
  - **Online**: continuous learning in a persistent world (good for "living" feel).
  - **Rollout**: auto-resets the world every 5 minutes, keeps learned weights (good for stable improvement).
- **Progress Dashboard**: four mini time-series charts (reward/min, novel interactions/min, tool clusters, prediction error) plus numeric readouts for training steps/min, loss (EMA), policy entropy (EMA), stall score, spawn throttle, and debris cleanup rate.
- **Milestones feed**: first composite, first station, first sustained yield, stall-break events, and other milestones appear live.
- **Anti-stall loop breakers**: a `StallDetector` monitors per-agent rolling windows for reward stagnation or repeated same-action loops and forces exploration when stalled (outside manufacture regime).
- **Population controller**: a `PopulationController` tracks spawn throttle and debris ratios to keep the world populated without infinite debris.
- **Prove Learning**: runs a deterministic before/after evaluation and exports a JSON report with measurable deltas (reward/min, novel/min, clusters, stall%).
- **Visual cues**: agent position (yellow diamond), target line, intent label (`forage | craft | explore | measure | idle`), and station overlays with quality readout.

### Sensible defaults

When you click **Start Evolution**, sensible defaults are applied:

| Parameter | Default |
|-----------|---------|
| Ticks/sec | 30 |
| Render every N ticks | 8 |
| trainEveryMs | 80 |
| batchSize | 32 |
| maxTrainMs/sec | 50 |
| Rolling record (sec) | 60 |

### Tests

- `npm test` includes unit tests for `StallDetector`, `PopulationController`, and `TrainingScheduler` in `test/evolution.test.ts`.

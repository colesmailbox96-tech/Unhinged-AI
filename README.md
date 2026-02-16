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

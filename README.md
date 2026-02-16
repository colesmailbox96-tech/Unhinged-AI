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

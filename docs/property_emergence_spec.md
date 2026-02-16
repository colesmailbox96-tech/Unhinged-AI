# Property Emergence Spec

- Objects are represented only by continuous `PropertyVector` values in `[0,1]`.
- Material generation samples seeded Gaussian distributions (stone-like, wood-like, plant-like, metal-like).
- Primitive verbs (`PICK_UP`, `DROP`, `MOVE_TO`, `STRIKE_WITH`, `BIND_TO`, `GRIND`, `HEAT`, `SOAK`) transform geometry and properties.
- `BIND` and `STRIKE` outcomes are continuous functions of properties and geometry. No recipe tables are used.
- Perception is partial/noisy. The perception head predicts hidden properties (hardness, brittleness, sharpness) from sensory channels and is updated only after interactions.
- RL uses a lightweight seeded epsilon-greedy bandit over primitive strategy choices, and reward is wood/minute (fracture count per simulated minute).
- Determinism is guaranteed by using the custom seeded RNG everywhere for spawn, noise, and policy exploration.

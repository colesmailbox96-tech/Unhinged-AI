/**
 * Needs & Metabolism — agents must maintain continuous internal state.
 *
 * Each agent has: energy, hydration, temperature, damage, fatigue
 * with simple homeostasis target ranges.
 *
 * Environmental hazards and scarcity pressure increase the urgency
 * to evolve and adapt — agents must discover these effects through
 * interaction.
 */

export interface AgentNeeds {
  energy: number;      // [0,1] - decreases with time + actions
  hydration: number;   // [0,1] - decreases over time; SOAK can increase
  temperature: number; // [0,1] normalized; 0.5 = ambient ideal
  damage: number;      // [0,1] - 0 = undamaged, 1 = dead
  fatigue: number;     // [0,1] - increases with heavy actions, decreases with REST
}

export interface NeedsConfig {
  energyDecayPerTick: number;
  hydrationDecayPerTick: number;
  temperatureDriftRate: number;   // drift toward 0.5 (ambient)
  fatigueDecayPerTick: number;    // recovery when resting
  deathThreshold: number;         // if any critical need exceeds/falls below this -> death
}

export const DEFAULT_NEEDS_CONFIG: NeedsConfig = {
  energyDecayPerTick: 0.003,
  hydrationDecayPerTick: 0.002,
  temperatureDriftRate: 0.01,
  fatigueDecayPerTick: 0.005,
  deathThreshold: 0.05,
};

export function createDefaultNeeds(): AgentNeeds {
  return {
    energy: 0.85,
    hydration: 0.8,
    temperature: 0.5,
    damage: 0,
    fatigue: 0.1,
  };
}

/** Environmental pressure modifiers applied each tick. */
export interface EnvironmentalPressure {
  /** Ambient temperature (0.5 = ideal, shifted by day/night and seasons). */
  ambientTemperature?: number;
  /** Extra hydration drain from drought events. */
  hydrationDrain?: number;
  /** Hazard exposure at the agent's position (0 = safe, higher = worse). */
  hazardExposure?: number;
  /** Hazard breakdown by kind (thermal, toxic, erosion). */
  hazardBreakdown?: { thermal: number; toxic: number; erosion: number };
  /** Scarcity multiplier (>= 1, grows over time). */
  scarcity?: number;
}

/** Action energy/fatigue costs by verb */
const ACTION_FATIGUE_COST: Record<string, number> = {
  STRIKE_WITH: 0.04,
  GRIND: 0.03,
  BIND_TO: 0.025,
  HEAT: 0.02,
  SOAK: 0.015,
  COOL: 0.015,
  ANCHOR: 0.025,
  CONTROL: 0.03,
  SHELTER: 0.005,
  MOVE_TO: 0.01,
  PICK_UP: 0.01,
  DROP: 0.005,
  REST: -0.03, // negative = recovery
};

/** Energy cost per action (for needs system) */
const NEEDS_ENERGY_COST: Record<string, number> = {
  STRIKE_WITH: 0.025,
  GRIND: 0.02,
  BIND_TO: 0.015,
  HEAT: 0.015,
  SOAK: 0.01,
  COOL: 0.012,
  ANCHOR: 0.015,
  CONTROL: 0.02,
  SHELTER: 0.003,
  MOVE_TO: 0.008,
  PICK_UP: 0.008,
  DROP: 0.004,
  REST: 0, // no cost
};

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/**
 * Update agent needs for one tick.
 * Returns updated needs and whether the agent should die.
 * Environmental pressure adds urgency — hazards damage the agent,
 * scarcity increases energy costs, and temperature shifts from day/night.
 */
export function tickNeeds(
  needs: AgentNeeds,
  action: string,
  cfg: NeedsConfig = DEFAULT_NEEDS_CONFIG,
  envPressure: EnvironmentalPressure = {},
): { needs: AgentNeeds; alive: boolean } {
  const scarcity = envPressure.scarcity ?? 1;
  const energyCost = ((NEEDS_ENERGY_COST[action] ?? 0.01) + cfg.energyDecayPerTick) * scarcity;
  const fatigueCost = ACTION_FATIGUE_COST[action] ?? 0;

  // Ambient temperature from environment (day/night + season)
  const ambientTemp = envPressure.ambientTemperature ?? 0.5;

  const updated: AgentNeeds = {
    energy: clamp01(needs.energy - energyCost),
    hydration: clamp01(needs.hydration - cfg.hydrationDecayPerTick - (envPressure.hydrationDrain ?? 0)),
    temperature: clamp01(needs.temperature + (ambientTemp - needs.temperature) * cfg.temperatureDriftRate),
    damage: clamp01(needs.damage), // damage doesn't auto-heal
    fatigue: clamp01(needs.fatigue + fatigueCost - (action === 'REST' ? cfg.fatigueDecayPerTick : 0)),
  };

  // SOAK increases hydration
  if (action === 'SOAK') {
    updated.hydration = clamp01(updated.hydration + 0.08);
  }
  // HEAT affects temperature
  if (action === 'HEAT') {
    updated.temperature = clamp01(updated.temperature + 0.06);
  }
  // COOL affects temperature
  if (action === 'COOL') {
    updated.temperature = clamp01(updated.temperature - 0.06);
  }
  // REST recovers fatigue and a bit of energy
  if (action === 'REST') {
    updated.energy = clamp01(updated.energy + 0.01);
  }

  // Hazard effects — agents must discover that being in hazard zones hurts them
  if (envPressure.hazardBreakdown) {
    const hb = envPressure.hazardBreakdown;
    // Thermal hazard shifts temperature away from ideal
    updated.temperature = clamp01(updated.temperature + hb.thermal * 0.08);
    // Toxic hazard causes damage over time
    updated.damage = clamp01(updated.damage + hb.toxic * 0.012);
    // Erosion hazard drains energy faster
    updated.energy = clamp01(updated.energy - hb.erosion * 0.006);
  } else if (envPressure.hazardExposure && envPressure.hazardExposure > 0) {
    // Fallback: generic hazard causes damage
    updated.damage = clamp01(updated.damage + envPressure.hazardExposure * 0.008);
  }

  // Death check: energy too low OR damage too high
  const alive = updated.energy > cfg.deathThreshold && updated.damage < (1 - cfg.deathThreshold);

  return { needs: updated, alive };
}

/**
 * Homeostasis reward: reward for maintaining needs in safe ranges.
 * Returns a value in [-0.5, 0.5] where positive means stable.
 */
export function homeostasisReward(needs: AgentNeeds): number {
  // Ideal ranges
  const energyScore = needs.energy > 0.3 ? 0.1 : (needs.energy - 0.3) * 0.5;
  const hydrationScore = needs.hydration > 0.25 ? 0.1 : (needs.hydration - 0.25) * 0.5;
  const tempScore = 1 - Math.abs(needs.temperature - 0.5) * 2; // best at 0.5
  const fatigueScore = needs.fatigue < 0.7 ? 0.1 : (0.7 - needs.fatigue) * 0.3;
  const damageScore = needs.damage < 0.3 ? 0.1 : (0.3 - needs.damage) * 0.3;

  return (energyScore + hydrationScore + tempScore * 0.1 + fatigueScore + damageScore) / 5;
}

/**
 * Determine the most urgent need for intent arbitration.
 */
export function mostUrgentNeed(needs: AgentNeeds): { need: string; urgency: number } {
  const urgencies = [
    { need: 'energy', urgency: Math.max(0, 0.4 - needs.energy) },
    { need: 'hydration', urgency: Math.max(0, 0.35 - needs.hydration) },
    { need: 'fatigue', urgency: Math.max(0, needs.fatigue - 0.7) },
    { need: 'temperature', urgency: Math.abs(needs.temperature - 0.5) > 0.3 ? Math.abs(needs.temperature - 0.5) : 0 },
    { need: 'damage', urgency: needs.damage > 0.5 ? needs.damage : 0 },
  ];
  urgencies.sort((a, b) => b.urgency - a.urgency);
  return urgencies[0];
}

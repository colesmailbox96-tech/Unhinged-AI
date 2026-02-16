/**
 * PopulationController — feedback controller that keeps world populations
 * in target bands to prevent debris blobs or target extinction.
 *
 * - targetsAlive kept in [minTargets, maxTargets]
 * - objectsTotal kept in [minObjects, maxObjects]
 * - fragments ratio bounded
 */

export interface PopulationBands {
  minTargets: number;
  maxTargets: number;
  minObjects: number;
  maxObjects: number;
  maxFragmentRatio: number; // e.g. 0.40
}

export interface PopulationMetrics {
  spawnThrottle: number;       // 0–1; 0 = no spawning, 1 = full spawning
  debrisCleanupRate: number;   // items cleaned per call
  purgedNonDebrisPerMin: number; // MUST be 0
}

export interface PopulationState {
  targetsAlive: number;
  objectsTotal: number;
  fragmentsTotal: number;
}

const DEFAULT_BANDS: PopulationBands = {
  minTargets: 10,
  maxTargets: 40,
  minObjects: 20,
  maxObjects: 400,
  maxFragmentRatio: 0.40,
};

export class PopulationController {
  private readonly bands: PopulationBands;
  private _spawnThrottle = 1;
  private _debrisCleanupRate = 0;
  private _purgedNonDebris = 0;

  constructor(bands: Partial<PopulationBands> = {}) {
    this.bands = { ...DEFAULT_BANDS, ...bands };
  }

  /**
   * Returns spawn probability modifier (0–1) and number of debris items to clean.
   */
  evaluate(state: PopulationState): { spawnProbability: number; debrisToClean: number } {
    // Spawn throttle: ramp down as we approach max
    const objFraction = state.objectsTotal / this.bands.maxObjects;
    const targetFraction = state.targetsAlive / this.bands.maxTargets;
    this._spawnThrottle = Math.max(0, Math.min(1, 1 - Math.max(objFraction, targetFraction)));

    // If targets below minimum, force spawning
    if (state.targetsAlive < this.bands.minTargets) {
      this._spawnThrottle = 1;
    }

    // Debris cleanup: if objects exceed max or fragment ratio too high
    let debrisToClean = 0;
    const fragmentRatio = state.objectsTotal > 0 ? state.fragmentsTotal / state.objectsTotal : 0;
    if (state.objectsTotal > this.bands.maxObjects) {
      debrisToClean = Math.min(state.fragmentsTotal, state.objectsTotal - this.bands.maxObjects);
    } else if (fragmentRatio > this.bands.maxFragmentRatio) {
      debrisToClean = Math.max(0, Math.ceil(state.fragmentsTotal - this.bands.maxFragmentRatio * state.objectsTotal));
    }
    this._debrisCleanupRate = debrisToClean;

    return { spawnProbability: this._spawnThrottle, debrisToClean };
  }

  metrics(): PopulationMetrics {
    return {
      spawnThrottle: this._spawnThrottle,
      debrisCleanupRate: this._debrisCleanupRate,
      purgedNonDebrisPerMin: this._purgedNonDebris, // always 0
    };
  }
}

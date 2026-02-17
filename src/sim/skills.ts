/**
 * Skill Discovery — tracks when agents reliably cause specific property transformations.
 * A skill is "owned" when success rate > threshold over last K trials.
 */

export interface SkillRecord {
  name: string;
  verb: string;
  property: string;           // property key that changes
  direction: 'increase' | 'decrease';
  trials: number;
  successes: number;
  successRate: number;
  owned: boolean;              // success rate > threshold
  discoveredAtTick?: number;
}

export interface SkillDiscoveryMetrics {
  totalSkills: number;
  ownedSkills: number;
  recentDiscoveries: string[];
  skillReward: number;
}

const SKILL_THRESHOLD = 0.6;        // success rate needed to "own" a skill
const MIN_TRIALS = 5;               // minimum trials before checking ownership
const MAX_TRIAL_WINDOW = 30;        // rolling window of trials
const NEW_SKILL_REWARD = 0.15;      // reward for discovering a new skill
const RELIABILITY_REWARD = 0.03;    // small reward for improving existing skills
const DIMINISHING_FACTOR = 0.7;     // diminishing returns on repeated skill discoveries

export class SkillTracker {
  private readonly skills = new Map<string, SkillRecord>();
  private readonly recentDiscoveries: Array<{ name: string; tick: number }> = [];
  private readonly trialHistory = new Map<string, boolean[]>();
  
  /**
   * Record a property transformation attempt.
   * @param verb - action verb used
   * @param property - property key that was targeted
   * @param beforeValue - property value before action
   * @param afterValue - property value after action
   * @param tick - current tick
   * @param deltaThreshold - minimum change to count as "success"
   */
  recordTransformation(
    verb: string,
    property: string,
    beforeValue: number,
    afterValue: number,
    tick: number,
    deltaThreshold = 0.01,
  ): number {
    const delta = afterValue - beforeValue;
    if (Math.abs(delta) < deltaThreshold) return 0;
    
    const direction: 'increase' | 'decrease' = delta > 0 ? 'increase' : 'decrease';
    const key = `${verb}:${property}:${direction}`;
    const name = `${verb} ${direction}s ${property}`;
    
    // Get or create skill record
    let skill = this.skills.get(key);
    if (!skill) {
      skill = {
        name,
        verb,
        property,
        direction,
        trials: 0,
        successes: 0,
        successRate: 0,
        owned: false,
      };
      this.skills.set(key, skill);
    }
    
    // Track trial in rolling window
    let history = this.trialHistory.get(key);
    if (!history) {
      history = [];
      this.trialHistory.set(key, history);
    }
    const success = Math.abs(delta) >= deltaThreshold;
    history.push(success);
    while (history.length > MAX_TRIAL_WINDOW) history.shift();
    
    skill.trials = history.length;
    skill.successes = history.filter(Boolean).length;
    skill.successRate = skill.successes / Math.max(1, skill.trials);
    
    // Check if skill is newly owned
    const wasOwned = skill.owned;
    skill.owned = skill.trials >= MIN_TRIALS && skill.successRate >= SKILL_THRESHOLD;
    
    let reward = 0;
    if (skill.owned && !wasOwned) {
      // New skill discovery!
      skill.discoveredAtTick = tick;
      const discoveryIndex = this.recentDiscoveries.length;
      this.recentDiscoveries.push({ name, tick });
      // Diminishing returns for subsequent discoveries
      reward = NEW_SKILL_REWARD * Math.pow(DIMINISHING_FACTOR, discoveryIndex);
    } else if (skill.owned && success) {
      // Small reward for maintaining/improving existing skills
      reward = RELIABILITY_REWARD * Math.pow(DIMINISHING_FACTOR, Math.max(0, skill.trials - MIN_TRIALS));
    }
    
    return reward;
  }
  
  /**
   * Get all discovered skills.
   */
  allSkills(): SkillRecord[] {
    return [...this.skills.values()];
  }
  
  /**
   * Get owned skills only.
   */
  ownedSkills(): SkillRecord[] {
    return [...this.skills.values()].filter(s => s.owned);
  }
  
  /**
   * Get metrics for display.
   */
  metrics(): SkillDiscoveryMetrics {
    const owned = this.ownedSkills();
    return {
      totalSkills: this.skills.size,
      ownedSkills: owned.length,
      recentDiscoveries: this.recentDiscoveries.slice(-5).map(d => d.name),
      skillReward: owned.reduce((sum, s) => sum + s.successRate * RELIABILITY_REWARD, 0),
    };
  }
}

export interface ToolEffectVector {
  damage: number;
  toolWear: number;
  fragments: number;
  propertyChanges: number;
}

function normalize(effect: ToolEffectVector): [number, number, number, number] {
  const raw: [number, number, number, number] = [effect.damage, effect.toolWear, effect.fragments, effect.propertyChanges];
  const mag = Math.hypot(...raw) || 1;
  return [raw[0] / mag, raw[1] / mag, raw[2] / mag, raw[3] / mag];
}

function cosine(a: [number, number, number, number], b: [number, number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function smooth(existing: [number, number, number, number], incoming: [number, number, number, number], keep = 0.7): [number, number, number, number] {
  const learn = 1 - keep;
  return [
    existing[0] * keep + incoming[0] * learn,
    existing[1] * keep + incoming[1] * learn,
    existing[2] * keep + incoming[2] * learn,
    existing[3] * keep + incoming[3] * learn,
  ];
}

export class ToolEmbedding {
  private readonly vectors = new Map<number, [number, number, number, number]>();
  private readonly interactionBins = new Set<string>();

  update(toolId: number, effect: ToolEffectVector): void {
    const incoming = normalize(effect);
    const existing = this.vectors.get(toolId);
    if (!existing) {
      this.vectors.set(toolId, incoming);
    } else {
      this.vectors.set(toolId, smooth(existing, incoming));
    }

    const q = (v: number): string => Math.round(v * 4).toString(10);
    this.interactionBins.add(`${q(effect.damage)}:${q(effect.toolWear)}:${q(effect.fragments)}:${q(effect.propertyChanges)}`);
  }

  similarity(toolA: number, toolB: number): number {
    const a = this.vectors.get(toolA);
    const b = this.vectors.get(toolB);
    if (!a || !b) return 0;
    return cosine(a, b);
  }

  clusterCount(similarityThreshold = 0.92): number {
    const vectors = [...this.vectors.values()];
    if (vectors.length === 0) return 0;
    const centers: [number, number, number, number][] = [];
    for (const vec of vectors) {
      const similar = centers.some((center) => cosine(vec, center) >= similarityThreshold);
      if (!similar) centers.push(vec);
    }
    return centers.length;
  }

  novelInteractionCount(): number {
    return this.interactionBins.size;
  }
}

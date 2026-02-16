import type { ObjID, Vec2 } from './object_model';
import type { World } from './world';

export interface WorksetState {
  ids: ObjID[];
  homeStationId?: ObjID;
  dropZone?: Vec2;
  ageSec: number;
  pinned: boolean;
}

export interface WorksetConfig {
  minSize: number;
  maxSize: number;
  preferredSize: number;
  refreshTtlSec: number;
  stationRadius: number;
  dropZoneOffsetX: number;
}

export const DEFAULT_WORKSET_CONFIG: WorksetConfig = {
  minSize: 3,
  maxSize: 12,
  preferredSize: 3,
  refreshTtlSec: 120,
  stationRadius: 1.1,
  dropZoneOffsetX: 0.45,
};

export function createEmptyWorkset(): WorksetState {
  return { ids: [], ageSec: 0, pinned: false };
}

function nearestStation(world: World, config: WorksetConfig): { id: ObjID; dropZone: Vec2 } | undefined {
  let best: { id: ObjID; dist: number; dropZone: Vec2 } | undefined;
  for (const station of world.stations.values()) {
    const dist = Math.hypot(world.agent.pos.x - station.worldPos.x, world.agent.pos.y - station.worldPos.y);
    if (!best || dist < best.dist) {
      best = {
        id: station.objectId,
        dist,
        dropZone: { x: Math.max(0, Math.min(world.width, station.worldPos.x + config.dropZoneOffsetX)), y: station.worldPos.y },
      };
    }
  }
  return best ? { id: best.id, dropZone: best.dropZone } : undefined;
}

function candidateIds(world: World): ObjID[] {
  return [...world.objects.values()]
    .filter((obj) => !obj.anchored && obj.heldBy === undefined && obj.integrity > 0.08)
    .sort((a, b) => {
      const qa = a.latentPrecision.surface_planarity + a.latentPrecision.microstructure_order + (1 - a.latentPrecision.impurity_level);
      const qb = b.latentPrecision.surface_planarity + b.latentPrecision.microstructure_order + (1 - b.latentPrecision.impurity_level);
      return qb - qa;
    })
    .map((obj) => obj.id);
}

export function refreshWorkset(world: World, state: WorksetState, config: WorksetConfig, dtSec: number): WorksetState {
  const next: WorksetState = {
    ...state,
    ids: state.ids.filter((id) => {
      const obj = world.objects.get(id);
      return Boolean(obj && obj.integrity > 0.08);
    }),
    ageSec: state.ageSec + Math.max(0, dtSec),
  };
  const station = nearestStation(world, config);
  if (station) {
    next.homeStationId = station.id;
    next.dropZone = station.dropZone;
  } else {
    next.homeStationId = undefined;
    next.dropZone = undefined;
  }
  const sizeFloor = Math.max(config.minSize, Math.min(config.maxSize, config.preferredSize));
  const shouldRefresh = !next.pinned && (next.ids.length < sizeFloor || next.ageSec >= config.refreshTtlSec);
  if (shouldRefresh) {
    const ids = candidateIds(world);
    next.ids = ids.slice(0, Math.max(config.minSize, Math.min(config.maxSize, config.preferredSize)));
    if (next.ids.length < config.minSize) {
      for (const obj of world.objects.values()) {
        if (next.ids.includes(obj.id) || obj.heldBy !== undefined || obj.integrity <= 0.08) continue;
        next.ids.push(obj.id);
        if (next.ids.length >= config.minSize) break;
      }
    }
    next.ageSec = 0;
  } else if (next.ids.length > config.maxSize) {
    next.ids = next.ids.slice(0, config.maxSize);
  }
  return next;
}

export function worksetAtStationFraction(world: World, state: WorksetState, radius: number): number {
  if (!state.dropZone || !state.ids.length) return 0;
  const near = state.ids.reduce((count, id) => {
    const obj = world.objects.get(id);
    if (!obj) return count;
    return count + (Math.hypot(obj.pos.x - state.dropZone!.x, obj.pos.y - state.dropZone!.y) <= radius ? 1 : 0);
  }, 0);
  return near / Math.max(1, state.ids.length);
}

export function avgDistanceToStation(world: World, state: WorksetState): number {
  if (!state.dropZone || !state.ids.length) return 0;
  const total = state.ids.reduce((sum, id) => {
    const obj = world.objects.get(id);
    if (!obj) return sum;
    return sum + Math.hypot(obj.pos.x - state.dropZone!.x, obj.pos.y - state.dropZone!.y);
  }, 0);
  return total / Math.max(1, state.ids.length);
}

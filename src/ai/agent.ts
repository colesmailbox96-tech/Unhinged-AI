import { PerceptionHead } from './perception';
import { World } from '../sim/world';
import type { ObjID, WorldObject } from '../sim/object_model';

export type Strategy = 'RANDOM_STRIKE' | 'BIND_THEN_STRIKE';

export interface EpisodeResult {
  woodGained: number;
  woodPerMinute: number;
  hardnessMaeBefore: number;
  hardnessMaeAfter: number;
  logs: string[];
}

function chooseByPerception(world: World, perception: PerceptionHead, avoid: ObjID[] = []): WorldObject[] {
  const hidden = world.getNearbyObjectIds().filter((id) => !avoid.includes(id)).map((id) => world.objects.get(id)).filter((obj): obj is WorldObject => Boolean(obj));

  return hidden.sort((a, b) => {
    const oa = perception.observe(a, world.rng);
    const ob = perception.observe(b, world.rng);
    const pa = perception.predict(oa);
    const pb = perception.predict(ob);
    const sa = pa.hardness * 0.25 + oa.observed_length * oa.observed_mass_estimate * (1.2 - oa.visual_symmetry) + oa.contact_area_estimate * 0.1;
    const sb = pb.hardness * 0.25 + ob.observed_length * ob.observed_mass_estimate * (1.2 - ob.visual_symmetry) + ob.contact_area_estimate * 0.1;
    return sb - sa;
  });
}

export function runEpisode(seed: number, strategy: Strategy, perception = new PerceptionHead(seed + 77), steps = 35): EpisodeResult {
  const world = new World(seed);
  const initialObjects = [...world.objects.values()];
  const hardnessMaeBefore = perception.hardnessError(initialObjects, world.rng.clone());

  const pickables = chooseByPerception(world, perception);
  if (pickables[0]) world.apply({ type: 'PICK_UP', objId: pickables[0].id });

  if (strategy === 'BIND_THEN_STRIKE' && pickables[1]) {
    world.apply({ type: 'BIND_TO', objId: pickables[1].id });
  }

  for (let i = 0; i < steps; i++) {
    if (strategy === 'RANDOM_STRIKE') {
      const nearby = world.getNearbyObjectIds().filter((id) => id !== world.getTargetId());
      const randomId = nearby[world.rng.int(0, Math.max(1, Math.floor(nearby.length * 0.6)))];
      if (randomId) world.apply({ type: 'PICK_UP', objId: randomId });
    }

    const targetId = world.getTargetId();
    if (!targetId) continue;

    if (i % 6 === 0 && strategy === 'BIND_THEN_STRIKE') {
      const candidates = chooseByPerception(world, perception, [world.agent.heldObjectId ?? -1]);
      if (candidates[0] && world.agent.heldObjectId) world.apply({ type: 'BIND_TO', objId: candidates[0].id });
    }

    if (i % 4 === 0) {
      const abrasive = chooseByPerception(world, perception, [world.agent.heldObjectId ?? -1])[0];
      if (abrasive) world.apply({ type: 'GRIND', abrasiveId: abrasive.id });
    }

    world.apply({ type: 'STRIKE_WITH', targetId });

    const held = world.agent.heldObjectId ? world.objects.get(world.agent.heldObjectId) : undefined;
    if (held) {
      const obs = perception.observe(held, world.rng);
      perception.train(obs, held.props, 1);
    }
  }

  const hardnessMaeAfter = perception.hardnessError([...world.objects.values()], world.rng.clone());
  return {
    woodGained: world.woodGained,
    woodPerMinute: world.woodGained / (steps / 60),
    hardnessMaeBefore,
    hardnessMaeAfter,
    logs: world.logs.slice(0, 25),
  };
}

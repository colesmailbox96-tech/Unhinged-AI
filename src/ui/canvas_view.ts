import { PerceptionHead } from '../ai/perception';
import type { WorldObject } from '../sim/object_model';
import { PROPERTY_KEYS } from '../sim/properties';
import { World } from '../sim/world';

export class CanvasView {
  selectedId?: number;
  private readonly canvas: HTMLCanvasElement;
  private readonly world: World;
  private readonly perception: PerceptionHead;

  constructor(canvas: HTMLCanvasElement, world: World, perception: PerceptionHead) {
    this.canvas = canvas;
    this.world = world;
    this.perception = perception;
    this.canvas.addEventListener('click', (event) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = ((event.clientX - rect.left) / rect.width) * this.world.width;
      const y = ((event.clientY - rect.top) / rect.height) * this.world.height;
      let best: { id: number; dist: number } | undefined;
      for (const obj of this.world.objects.values()) {
        const dist = Math.hypot(obj.pos.x - x, obj.pos.y - y);
        if (!best || dist < best.dist) best = { id: obj.id, dist };
      }
      this.selectedId = best?.id;
    });
  }

  private colorOf(obj: WorldObject): string {
    if (obj.debugFamily === 'target-visual') return '#79c26f';
    if (obj.constituents) return '#d6ad4f';
    return '#7eb2ff';
  }

  render(selectedEl: HTMLElement, logEl: HTMLElement): void {
    const ctx = this.canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const sx = this.canvas.width / this.world.width;
    const sy = this.canvas.height / this.world.height;

    for (const obj of this.world.objects.values()) {
      const x = obj.pos.x * sx;
      const y = obj.pos.y * sy;
      const radius = Math.max(3, obj.radius * sx);
      const line = obj.length * sx * 0.4;

      ctx.strokeStyle = this.colorOf(obj);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - line * 0.5, y);
      ctx.lineTo(x + line * 0.5, y);
      ctx.stroke();

      ctx.fillStyle = this.colorOf(obj);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (this.selectedId === obj.id) {
        ctx.strokeStyle = '#fff';
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    const selected = this.selectedId ? this.world.objects.get(this.selectedId) : undefined;
    if (selected) {
      const obs = this.perception.observe(selected, this.world.rng);
      const pred = this.perception.predict(obs);
      const trueVec = PROPERTY_KEYS.map((k) => `${k}: ${selected.props[k].toFixed(2)}`).join(', ');
      selectedEl.innerHTML = `selected: ${selected.id}<br/>true: ${trueVec}<br/>perceived: mass=${obs.massish.toFixed(2)} rough=${obs.roughnessish.toFixed(2)} length=${obs.lengthish.toFixed(2)}<br/>pred hidden: hard=${pred.hardness.toFixed(2)} brit=${pred.brittleness.toFixed(2)} sharp=${pred.sharpness.toFixed(2)} ±${pred.uncertainty.toFixed(2)}`;
    } else {
      selectedEl.textContent = 'selected: none';
    }

    logEl.textContent = this.world.logs.slice(0, 18).join('\n');
  }
}

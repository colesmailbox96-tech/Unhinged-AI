import { PerceptionHead } from '../ai/perception';
import type { WorldObject } from '../sim/object_model';
import { PROPERTY_KEYS } from '../sim/properties';
import { World } from '../sim/world';

export class CanvasView {
  selectedId?: number;
  showTrueLatentState = false;
  showWorkset = true;
  agentIntent = 'idle';
  agentTargetId?: number;
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

    if (this.world.lastStrikeArc && this.world.lastStrikeArc.alpha > 0.02) {
      const arc = this.world.lastStrikeArc;
      ctx.strokeStyle = `rgba(255, 234, 128, ${arc.alpha.toFixed(3)})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(arc.center.x * sx, arc.center.y * sy, arc.radius * sx, arc.start, arc.end);
      ctx.stroke();
      arc.alpha *= 0.92;
    }

    if (this.world.predictedStrikeArc && this.world.predictedStrikeArc.alpha > 0.02) {
      const arc = this.world.predictedStrikeArc;
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = `rgba(106, 235, 255, ${arc.alpha.toFixed(3)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(arc.center.x * sx, arc.center.y * sy, arc.radius * sx, arc.start, arc.end);
      ctx.stroke();
      ctx.restore();
      arc.alpha *= 0.92;
    }

    for (const obj of this.world.objects.values()) {
      const x = obj.pos.x * sx;
      const y = obj.pos.y * sy;
      const radius = Math.max(3, obj.radius * sx);
      const line = obj.length * sx * 0.4;

      ctx.strokeStyle = this.colorOf(obj);
      ctx.lineWidth = 2;
      if (obj.shapeType === 'rod') {
        ctx.beginPath();
        ctx.moveTo(x - line * 0.5, y);
        ctx.lineTo(x + line * 0.5, y);
        ctx.stroke();
      } else if (obj.shapeType === 'shard') {
        const h = Math.max(4, obj.thickness * sy * 0.8);
        ctx.beginPath();
        ctx.moveTo(x - line * 0.5, y + h);
        ctx.lineTo(x + line * 0.5, y);
        ctx.lineTo(x - line * 0.2, y - h);
        ctx.closePath();
        ctx.stroke();
      } else if (obj.shapeType === 'plate') {
        const w = Math.max(6, line);
        const h = Math.max(4, obj.thickness * sy);
        ctx.strokeRect(x - w * 0.5, y - h * 0.5, w, h);
      }

      ctx.fillStyle = this.colorOf(obj);
      if (obj.shapeType === 'sphere') {
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      } else if (obj.shapeType === 'plate') {
        const w = Math.max(6, line);
        const h = Math.max(4, obj.thickness * sy);
        ctx.fillRect(x - w * 0.5, y - h * 0.5, w, h);
      } else {
        ctx.beginPath();
        ctx.arc(x + line * 0.5, y, Math.max(2, radius * 0.45), 0, Math.PI * 2);
        ctx.fill();
      }

      if (obj.constituents) {
        ctx.strokeStyle = '#ffe3a8';
        ctx.beginPath();
        ctx.moveTo(x - line * 0.4, y);
        ctx.lineTo(x + line * 0.4, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x + line * 0.42, y, Math.max(2, radius * 0.35), 0, Math.PI * 2);
        ctx.stroke();
      }

      if (this.selectedId === obj.id) {
        ctx.strokeStyle = '#fff';
        ctx.beginPath();
        ctx.arc(x, y, radius + 4, 0, Math.PI * 2);
        ctx.stroke();
      }
      if (this.showWorkset && this.world.worksetDebugIds.includes(obj.id)) {
        ctx.strokeStyle = '#ff7f7f';
        ctx.beginPath();
        ctx.arc(x, y, radius + 3, 0, Math.PI * 2);
        ctx.stroke();
        if (this.world.worksetDropZone) {
          ctx.strokeStyle = 'rgba(255,127,127,0.35)';
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(this.world.worksetDropZone.x * sx, this.world.worksetDropZone.y * sy);
          ctx.stroke();
        }
      }
    }
    if (this.showWorkset && this.world.worksetDropZone) {
      ctx.strokeStyle = '#ff7f7f';
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(this.world.worksetDropZone.x * sx, this.world.worksetDropZone.y * sy, 12, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (this.world.predictionRealityOverlay) {
      const overlay = this.world.predictionRealityOverlay;
      ctx.fillStyle = '#d8fbff';
      ctx.font = '12px monospace';
      ctx.fillText(`pred dmg ${overlay.predicted.damage.toFixed(2)} wear ${overlay.predicted.toolWear.toFixed(2)} frags ${overlay.predicted.fragments.toFixed(2)}`, 10, 18);
      ctx.fillText(`real dmg ${overlay.actual.damage.toFixed(2)} wear ${overlay.actual.toolWear.toFixed(2)} frags ${overlay.actual.fragments.toFixed(2)}`, 10, 34);
      ctx.fillText(`|error| dmg ${overlay.error.damage.toFixed(2)} wear ${overlay.error.toolWear.toFixed(2)} frags ${overlay.error.fragments.toFixed(2)}`, 10, 50);
    } else if (this.world.predictedStrikeDamage !== undefined && this.world.actualStrikeDamage !== undefined) {
      ctx.fillStyle = '#d8fbff';
      ctx.font = '12px monospace';
      ctx.fillText(`pred damage ${this.world.predictedStrikeDamage.toFixed(2)} | actual ${this.world.actualStrikeDamage.toFixed(2)}`, 10, 18);
    }

    const selected = this.selectedId ? this.world.objects.get(this.selectedId) : undefined;
    if (selected) {
      const obs = this.perception.observe(selected, this.world.rng);
      const pred = this.perception.predict(obs);
      const trueVec = PROPERTY_KEYS.map((k) => `${k}: ${selected.props[k].toFixed(2)}`).join(', ');
      const latent = selected.latentPrecision;
      selectedEl.innerHTML = [
        `selected: ${selected.id}`,
        `true: ${trueVec}`,
        this.showTrueLatentState
          ? `latent(debug): planarity=${latent.surface_planarity.toFixed(2)} impurity=${latent.impurity_level.toFixed(2)} order=${latent.microstructure_order.toFixed(2)} stress=${latent.internal_stress.toFixed(2)} resolution=${latent.feature_resolution_limit.toFixed(2)}`
          : 'latent(debug): hidden',
        `perceived: length=${obs.observed_length.toFixed(2)} mass≈${obs.observed_mass_estimate.toFixed(2)} symmetry=${obs.visual_symmetry.toFixed(2)} contact≈${obs.contact_area_estimate.toFixed(2)} tex≈${obs.texture_proxy.toFixed(2)} feedback≈${obs.interaction_feedback_history.toFixed(2)}`,
        `pred hidden: hard=${pred.hardness.toFixed(2)} brit=${pred.brittleness.toFixed(2)} sharp=${pred.sharpness.toFixed(2)} ±${pred.uncertainty.toFixed(2)}`,
      ].join('<br/>');
    } else {
      selectedEl.textContent = 'selected: none';
    }

    // Agent target line + intent label
    const agentX = this.world.agent.pos.x * sx;
    const agentY = this.world.agent.pos.y * sy;
    // Draw agent as small diamond
    ctx.fillStyle = '#ff0';
    ctx.beginPath();
    ctx.moveTo(agentX, agentY - 5);
    ctx.lineTo(agentX + 4, agentY);
    ctx.lineTo(agentX, agentY + 5);
    ctx.lineTo(agentX - 4, agentY);
    ctx.closePath();
    ctx.fill();
    // Draw line to current target
    if (this.agentTargetId) {
      const tgt = this.world.objects.get(this.agentTargetId);
      if (tgt) {
        ctx.strokeStyle = 'rgba(255,255,0,0.4)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(agentX, agentY);
        ctx.lineTo(tgt.pos.x * sx, tgt.pos.y * sy);
        ctx.stroke();
      }
    }
    // Draw agent intent label
    ctx.fillStyle = 'rgba(255,255,200,0.8)';
    ctx.font = '10px system-ui';
    ctx.fillText(this.agentIntent, agentX + 6, agentY - 6);
    // Station overlays
    for (const station of this.world.stations.values()) {
      const stX = station.worldPos.x * sx;
      const stY = station.worldPos.y * sy;
      ctx.strokeStyle = 'rgba(200,150,255,0.5)';
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(stX, stY, 18, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(200,150,255,0.7)';
      ctx.font = '9px system-ui';
      ctx.fillText(`stn q=${station.quality.toFixed(2)}`, stX + 10, stY - 4);
    }

    logEl.textContent = this.world.logs.slice(0, 18).join('\n');
  }
}

/**
 * SpatialGrid — grid-based spatial hashing for fast proximity queries.
 *
 * Replaces O(N) brute-force search in getNearbyObjectIds() with
 * O(K) lookups where K is the number of objects in nearby cells.
 */

export class SpatialGrid<T extends { id: number; pos: { x: number; y: number } }> {
  private readonly cellSize: number;
  private readonly cells = new Map<string, T[]>();
  private readonly itemCells = new Map<number, string>(); // id → cell key

  constructor(cellSize = 2.0) {
    this.cellSize = cellSize;
  }

  private keyFor(x: number, y: number): string {
    const cx = Math.floor(x / this.cellSize);
    const cy = Math.floor(y / this.cellSize);
    return `${cx},${cy}`;
  }

  /** Insert or update an item in the grid. */
  upsert(item: T): void {
    const newKey = this.keyFor(item.pos.x, item.pos.y);
    const oldKey = this.itemCells.get(item.id);

    if (oldKey === newKey) return; // no cell change

    // Remove from old cell
    if (oldKey !== undefined) {
      const oldCell = this.cells.get(oldKey);
      if (oldCell) {
        const idx = oldCell.findIndex(i => i.id === item.id);
        if (idx >= 0) oldCell.splice(idx, 1);
        if (oldCell.length === 0) this.cells.delete(oldKey);
      }
    }

    // Insert into new cell
    let newCell = this.cells.get(newKey);
    if (!newCell) {
      newCell = [];
      this.cells.set(newKey, newCell);
    }
    newCell.push(item);
    this.itemCells.set(item.id, newKey);
  }

  /** Remove an item from the grid. */
  remove(id: number): void {
    const key = this.itemCells.get(id);
    if (key === undefined) return;
    const cell = this.cells.get(key);
    if (cell) {
      const idx = cell.findIndex(i => i.id === id);
      if (idx >= 0) cell.splice(idx, 1);
      if (cell.length === 0) this.cells.delete(key);
    }
    this.itemCells.delete(id);
  }

  /** Find all items within `radius` of point (cx, cy). */
  queryRadius(cx: number, cy: number, radius: number): T[] {
    const result: T[] = [];
    const r2 = radius * radius;
    const minCx = Math.floor((cx - radius) / this.cellSize);
    const maxCx = Math.floor((cx + radius) / this.cellSize);
    const minCy = Math.floor((cy - radius) / this.cellSize);
    const maxCy = Math.floor((cy + radius) / this.cellSize);

    for (let gx = minCx; gx <= maxCx; gx++) {
      for (let gy = minCy; gy <= maxCy; gy++) {
        const cell = this.cells.get(`${gx},${gy}`);
        if (!cell) continue;
        for (const item of cell) {
          const dx = item.pos.x - cx;
          const dy = item.pos.y - cy;
          if (dx * dx + dy * dy <= r2) result.push(item);
        }
      }
    }
    return result;
  }

  /** Clear all entries. */
  clear(): void {
    this.cells.clear();
    this.itemCells.clear();
  }

  /** Number of tracked items. */
  get size(): number {
    return this.itemCells.size;
  }
}

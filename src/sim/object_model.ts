import type { PropertyVector } from './properties';

export type ObjID = number;

export interface WorldObject {
  id: ObjID;
  pos: { x: number; y: number };
  vel: { x: number; y: number };
  radius: number;
  length: number;
  props: PropertyVector;
  integrity: number;
  heldBy?: number;
  constituents?: ObjID[];
  debugFamily?: string;
}

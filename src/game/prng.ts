export type RandomState = readonly [number, number, number, number];

const rotateLeft = (value: number, shift: number): number =>
  ((value << shift) | (value >>> (32 - shift))) >>> 0;

const splitMix32 = (seed: number): (() => number) => {
  let value = seed >>> 0;
  return () => {
    value = (value + 0x9e3779b9) >>> 0;
    let mixed = value;
    mixed = Math.imul(mixed ^ (mixed >>> 16), 0x21f0aaad);
    mixed = Math.imul(mixed ^ (mixed >>> 15), 0x735a2d97);
    return (mixed ^ (mixed >>> 15)) >>> 0;
  };
};

export class Xoshiro128StarStar {
  static fromSeed(seed: number): Xoshiro128StarStar {
    const nextSeed = splitMix32(seed);
    const state: RandomState = [nextSeed(), nextSeed(), nextSeed(), nextSeed()];
    return new Xoshiro128StarStar(state.every((part) => part === 0) ? [1, 0, 0, 0] : state);
  }

  constructor(private state: RandomState) {}

  nextUint32(): number {
    const [state0, state1, state2, state3] = this.state;
    const result = Math.imul(rotateLeft(Math.imul(state1, 5) >>> 0, 7), 9) >>> 0;
    const shifted = (state1 << 9) >>> 0;
    const next2 = (state2 ^ state0) >>> 0;
    const next3 = (state3 ^ state1) >>> 0;
    const next1 = (state1 ^ next2) >>> 0;
    const next0 = (state0 ^ next3) >>> 0;

    this.state = [next0, next1, (next2 ^ shifted) >>> 0, rotateLeft(next3, 11)];
    return result;
  }

  nextFloat(): number {
    return this.nextUint32() / 0x1_0000_0000;
  }

  snapshot(): RandomState {
    return [...this.state];
  }
}

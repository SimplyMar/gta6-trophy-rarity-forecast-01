import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
const LAUNCH_MS = Date.parse('2026-11-19T00:00:00Z');

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: 'platinum' | 'gold' | 'silver' | 'bronze';
  points: number;
  secret: boolean;
}

interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: HistoryRead[];
}

const clamp = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => Math.log(clamp(pct) / (100 - clamp(pct)));
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

function priorShiftLogit(t: Trophy, horizonDays: number): number {
  const desc = `${t.name} ${t.description}`.toLowerCase();
  const scale = Math.min(1, horizonDays / 7);
  if (t.tier === 'platinum' || desc.includes('100% completion')) return -0.85 * scale;
  if (t.tier === 'gold' || desc.includes('final mission')) return -0.25 * scale;
  if (desc.includes('chapter 1') || desc.includes('gone fishing') || desc.includes('breakfast run') || desc.includes('cocktail')) {
    return 1.35 * scale;
  }
  if (desc.includes('chapter 2') || desc.includes('chapter 3') || desc.includes('chapter 4')) {
    return 0.65 * scale;
  }
  if (desc.includes('250') || desc.includes('100 ') || desc.includes('all ') || desc.includes('every ')) {
    return -0.55 * scale;
  }
  return 0.1 * scale;
}

function fitDampedLogTrend(points: Array<{ x: number; y: number }>, targetX: number): { pred: number; se: number } | null {
  if (points.length < 3) return null;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  const mx = sx / points.length;
  const my = sy / points.length;
  let sxx = 0;
  let sxy = 0;
  for (const p of points) {
    const dx = p.x - mx;
    sxx += dx * dx;
    sxy += dx * (p.y - my);
  }
  if (sxx < 1e-6) return null;
  const rawSlope = sxy / sxx;
  // Shrink slope toward 0 to prevent wild extrapolation from early refreshes
  const shrinkage = points.length / (points.length + 4);
  const slope = Math.max(-1.2, Math.min(1.5, rawSlope * shrinkage));
  const last = points[points.length - 1]!;
  const pred = last.y + slope * (targetX - last.x);

  let sse = 0;
  for (const p of points) {
    const est = my + slope * (p.x - mx);
    sse += (p.y - est) ** 2;
  }
  const residSd = Math.sqrt(sse / Math.max(1, points.length - 2));
  const se = Math.min(1.2, residSd * (1 + Math.abs(targetX - last.x) * 0.35));
  return { pred, se };
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();

for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const last = c.history[c.history.length - 1];
  if (!last) throw new Error(`${file}: empty history`);

  // Deduplicate history to distinct Exophase refresh snapshots
  const refreshes: HistoryRead[] = [];
  for (const h of c.history) {
    const prev = refreshes[refreshes.length - 1];
    if (!prev || prev.players_tracked !== h.players_tracked || prev.trophies_earned !== h.trophies_earned) {
      refreshes.push(h);
    }
  }

  const horizonDays = c.horizon_minutes / 1440;
  const targetHours = Math.max(0.1, (Date.parse(c.target) - LAUNCH_MS) / 3_600_000);
  const targetX = Math.log(1 + targetHours);

  const trophies = c.trophies.map((t, i) => {
    const lastRate = last.rates[i] ?? 0.5;
    const lastLogit = logit(lastRate);
    const pts: Array<{ x: number; y: number }> = [];
    for (const r of refreshes) {
      const val = r.rates[i];
      if (val !== null && val !== undefined) {
        const hrs = Math.max(0.1, (Date.parse(r.at) - LAUNCH_MS) / 3_600_000);
        pts.push({ x: Math.log(1 + hrs), y: logit(val) });
      }
    }

    const priorMed = lastLogit + priorShiftLogit(t, horizonDays);
    const trend = fitDampedLogTrend(pts, targetX);
    const medianLogit = trend ? 0.65 * trend.pred + 0.35 * priorMed : priorMed;

    // Wider calibrated uncertainty for short histories / refresh jumps
    const baseSigma = refreshes.length <= 2 ? 0.62 : 0.44;
    const trendSe = trend ? trend.se : 0.25 * Math.sqrt(horizonDays);
    const spread = Math.max(0.18, Math.sqrt((baseSigma * Math.sqrt(horizonDays)) ** 2 + trendSe ** 2));

    const q = Z.map((z) => round4(clamp(invLogit(medianLogit + z * spread))));
    for (let k = 1; k < q.length; k++) {
      if (q[k]! < q[k - 1]!) q[k] = q[k - 1]!;
    }
    return { id: t.id, q };
  });

  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
process.stdout.write(`forecasted ${inputs.length} case(s)\n`);

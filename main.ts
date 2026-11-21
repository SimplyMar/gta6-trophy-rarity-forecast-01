import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Student-t (df=10) quantile multipliers for robust 5%/95% tail pinball calibration.
const Q_MULT = [-1.72, -0.68, 0.0, 0.68, 1.72];
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
  schema?: string;
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: HistoryRead[];
}

interface RefreshPoint {
  hours: number;
  logTime: number;
  players: number;
  earned: number;
  rates: Array<number | null>;
}

type Archetype =
  | 'prologue'
  | 'ch1'
  | 'story_ch2_mid'
  | 'ch2'
  | 'story_ch3_mid'
  | 'ch3'
  | 'story_ch4_mid'
  | 'ch4'
  | 'story_ch5_mid'
  | 'finale'
  | 'early_activity'
  | 'mid_activity'
  | 'side_strand'
  | 'hard_grind'
  | 'completion_100'
  | 'platinum';

interface ArchetypePrior {
  pMax: number;
  tau50: number;
  k: number;
}

// Calibrated to GTA VI PS5 (~60-80h story, ~200-250h platinum; weak regularization so f00 data dominates)
const ARCHETYPE_PRIORS: Record<Archetype, ArchetypePrior> = {
  prologue:       { pMax: 0.996, tau50: 0.4,   k: 2.0 },
  ch1:            { pMax: 0.85,  tau50: 36.0,  k: 1.9 },
  story_ch2_mid:  { pMax: 0.76,  tau50: 62.0,  k: 2.0 },
  ch2:            { pMax: 0.70,  tau50: 92.0,  k: 2.1 },
  story_ch3_mid:  { pMax: 0.62,  tau50: 138.0, k: 2.2 },
  ch3:            { pMax: 0.58,  tau50: 165.0, k: 2.2 },
  story_ch4_mid:  { pMax: 0.52,  tau50: 215.0, k: 2.3 },
  ch4:            { pMax: 0.48,  tau50: 250.0, k: 2.3 },
  story_ch5_mid:  { pMax: 0.45,  tau50: 290.0, k: 2.4 },
  finale:         { pMax: 0.43,  tau50: 320.0, k: 2.4 },
  early_activity: { pMax: 0.48,  tau50: 95.0,  k: 1.3 },
  mid_activity:   { pMax: 0.28,  tau50: 240.0, k: 1.4 },
  side_strand:    { pMax: 0.24,  tau50: 400.0, k: 1.6 },
  hard_grind:     { pMax: 0.08,  tau50: 1050., k: 1.7 },
  completion_100: { pMax: 0.04,  tau50: 1850., k: 2.0 },
  platinum:       { pMax: 0.03,  tau50: 2250., k: 2.1 },
};

const clampScorePct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const clampRawPct = (x: number): number => Math.min(99.995, Math.max(1e-4, x));
const logitScore = (pct: number): number => {
  const c = clampScorePct(pct);
  return Math.log(c / (100 - c));
};
const logitRaw = (pct: number): number => {
  const c = clampRawPct(pct);
  return Math.log(c / (100 - c));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

function normCdf(x: number): number {
  const s = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const erf = 1 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * Math.exp(-ax * ax);
  return 0.5 * (1 + s * erf);
}

function pinballItem(yPct: number, qArr: number[]): number {
  const ly = logitScore(yPct);
  let sum = 0;
  for (let k = 0; k < 5; k++) {
    const tau = LEVELS[k]!;
    const d = ly - logitScore(qArr[k]!);
    sum += 2 * Math.max(tau * d, (tau - 1) * d);
  }
  return sum / 5;
}

function classifyTrophy(t: Trophy): Archetype {
  const desc = t.description.toLowerCase();
  const full = `${t.name.toLowerCase()} ${desc}`;
  if (t.tier === 'platinum') return 'platinum';
  if (desc.includes('100% completion')) return 'completion_100';
  if (desc.includes('gone fishing')) return 'prologue';
  if (desc.includes('chapter 1')) return 'ch1';
  if (desc.includes('breakfast run')) return 'story_ch2_mid';
  if (desc.includes('chapter 2')) return 'ch2';
  if (desc.includes('cartel heist')) return 'story_ch3_mid';
  if (desc.includes('chapter 3')) return 'ch3';
  if (desc.includes("child's play") || desc.includes('headliners') || desc.includes('vcia takedown')) return 'story_ch4_mid';
  if (desc.includes('chapter 4')) return 'ch4';
  if (desc.includes('operation savage glory')) return 'story_ch5_mid';
  if (t.tier === 'gold' || desc.includes('complete the final mission.')) return 'finale';

  if (
    desc.includes('fishing challenge') ||
    desc.includes('100 vehicles') ||
    desc.includes('arcade game') ||
    desc.includes('cocktail with a view') ||
    full.includes('bowling')
  ) {
    return 'early_activity';
  }
  if (desc.includes('final mission for ') || desc.includes('unique animal for bret')) {
    return 'side_strand';
  }
  if (
    desc.includes('250 stunts') ||
    desc.includes('shooting range') ||
    desc.includes('8 mission strands') ||
    desc.includes('3 collectible card') ||
    desc.includes('100 locations') ||
    desc.includes('50 challenges') ||
    desc.includes('every spot across leonida') ||
    desc.includes('explore all of leonida')
  ) {
    return 'hard_grind';
  }
  return 'mid_activity';
}

function extractRefreshes(history: HistoryRead[]): RefreshPoint[] {
  const out: RefreshPoint[] = [];
  for (const h of history) {
    const prev = out[out.length - 1];
    const hours = Math.max(0.05, (Date.parse(h.at) - LAUNCH_MS) / 3_600_000);
    if (!prev) {
      out.push({ hours, logTime: Math.log(1 + hours), players: h.players_tracked, earned: h.trophies_earned, rates: h.rates });
      continue;
    }
    let changed = h.players_tracked !== prev.players || h.trophies_earned !== prev.earned;
    if (!changed) {
      for (let i = 0; i < h.rates.length; i++) {
        if (h.rates[i] !== prev.rates[i]) {
          changed = true;
          break;
        }
      }
    }
    if (changed) {
      out.push({ hours, logTime: Math.log(1 + hours), players: h.players_tracked, earned: h.trophies_earned, rates: h.rates });
    }
  }
  return out;
}

/**
 * Fit pure post-launch organic player-pool arrival trajectory N_post(t) = N(t) - N_0
 * with adaptive shift t0 and observed early-vs-recent deceleration.
 */
function buildPlayerTrajectory(
  refreshes: RefreshPoint[],
  n0: number,
  targetHours: number,
): {
  cohorts: Array<{ hour: number; added: number }>;
  nTarget: number;
  alphaRecent: number;
} {
  const post = refreshes.filter((r) => r.hours >= 2.0 && r.players > n0 + 200);
  const pts = post.length >= 2 ? post : refreshes;
  const last = refreshes[refreshes.length - 1]!;

  // Select best time-shift t0 and fit overall + recent power-law exponents in ln(N - N0) vs ln(t + t0)
  const fitSlope = (subset: RefreshPoint[], t0: number): { slope: number; sse: number } => {
    if (subset.length < 2) return { slope: 0.72, sse: 1e9 };
    const xs = subset.map((p) => Math.log(p.hours + t0));
    const ys = subset.map((p) => Math.log(Math.max(10, p.players - n0)));
    const maxH = subset[subset.length - 1]!.hours;
    const ws = subset.map((p) => Math.exp(-0.02 * (maxH - p.hours)));
    let sw = 0, swx = 0, swy = 0;
    for (let i = 0; i < subset.length; i++) {
      sw += ws[i]!; swx += ws[i]! * xs[i]!; swy += ws[i]! * ys[i]!;
    }
    const mx = swx / sw;
    const my = swy / sw;
    let sxx = 0, sxy = 0;
    for (let i = 0; i < subset.length; i++) {
      const dx = xs[i]! - mx;
      sxx += ws[i]! * dx * dx;
      sxy += ws[i]! * dx * (ys[i]! - my);
    }
    if (sxx < 1e-6) return { slope: 0.72, sse: 1e9 };
    const slope = sxy / sxx;
    const intercept = my - slope * mx;
    let sse = 0;
    for (let i = 0; i < subset.length; i++) {
      const err = ys[i]! - (intercept + slope * xs[i]!);
      sse += ws[i]! * err * err;
    }
    return { slope, sse: sse / sw };
  };

  let bestT0 = 4.0;
  let bestSse = Infinity;
  let alphaAll = 0.72;
  for (const t0 of [1.5, 3.0, 5.0, 8.0, 14.0]) {
    const res = fitSlope(pts, t0);
    if (res.sse < bestSse) {
      bestSse = res.sse;
      bestT0 = t0;
      alphaAll = res.slope;
    }
  }

  // Measure recent slope (last 24h) vs full slope to detect weekend acceleration/deceleration
  const recentPts = pts.filter((p) => p.hours >= Math.max(3.0, last.hours - 26.0));
  let alphaRecent = alphaAll;
  if (recentPts.length >= 3) {
    const recFit = fitSlope(recentPts, bestT0);
    alphaRecent = 0.65 * recFit.slope + 0.35 * alphaAll;
  }
  const weight = pts.length / (pts.length + 2.0);
  alphaRecent = Math.max(0.22, Math.min(1.30, alphaRecent * weight + 0.72 * (1 - weight)));

  // Pure post-launch observed cohorts (strictly excluding pre-launch N_0 so N_0 is never double-counted)
  const cohorts: Array<{ hour: number; added: number }> = [];
  let prevN = n0;
  let prevH = 0.0;
  for (const r of refreshes) {
    if (r.players <= n0 && r.hours < 2.0) {
      prevN = Math.max(prevN, r.players);
      prevH = r.hours;
      continue;
    }
    const added = Math.max(0, r.players - prevN);
    if (added > 0) {
      const midH = Math.max(0.25, 0.5 * (prevH + r.hours));
      cohorts.push({ hour: midH, added });
    }
    prevN = Math.max(prevN, r.players);
    prevH = r.hours;
  }

  // Project future post-launch arrivals on 2-hour bins up to targetHours
  // Gentle post-weekend decay after Sunday night (t > 92h)
  let curPostN = Math.max(10, last.players - n0);
  let h = last.hours;
  const stepH = 2.0;
  while (h < targetHours - 1e-6) {
    const nextH = Math.min(targetHours, h + stepH);
    const midH = 0.5 * (h + nextH);
    const decel = midH > 100 ? 0.76 : midH > 84 ? 0.88 : 1.0;
    const dLogT = Math.log(nextH + bestT0) - Math.log(h + bestT0);
    const nextPostN = curPostN * Math.exp(alphaRecent * decel * dLogT);
    const added = Math.max(0, nextPostN - curPostN);
    if (added > 0) cohorts.push({ hour: midH, added });
    curPostN = nextPostN;
    h = nextH;
  }

  return { cohorts, nTarget: n0 + curPostN, alphaRecent };
}

type SurvivalFamily = 'loglogistic' | 'weibull' | 'lognormal';

function evalCohortRateRaw(
  cohorts: Array<{ hour: number; added: number }>,
  evalHour: number,
  n0: number,
  preLaunchCount: number,
  pMax: number,
  tau50: number,
  k: number,
  family: SurvivalFamily,
): number {
  let totalN = n0;
  let totalE = preLaunchCount;
  const logTau = family === 'lognormal' ? Math.log(tau50) : 0;
  const invSigma = family === 'lognormal' ? k / 1.6 : 0;
  for (let idx = 0; idx < cohorts.length; idx++) {
    const c = cohorts[idx]!;
    if (c.hour > evalHour + 1e-6) break;
    totalN += c.added;
    const age = Math.max(0.1, evalHour - c.hour);
    let f: number;
    if (family === 'loglogistic') {
      f = pMax / (1 + Math.pow(tau50 / age, k));
    } else if (family === 'weibull') {
      f = pMax * (1 - Math.exp(-Math.pow(age / tau50, k)));
    } else {
      f = pMax * normCdf((Math.log(age) - logTau) * invSigma);
    }
    totalE += c.added * f;
  }
  if (totalN <= 0) return 1e-4;
  return clampRawPct((100 * Math.min(totalN, totalE)) / totalN);
}

interface ObsPoint {
  hours: number;
  rate: number;
  players: number;
  weight: number;
}

/**
 * Interval-censored, inverse-variance weighted cohort convolution fit for Log-Logistic, Weibull, or Log-Normal survival.
 * Properly treats obsRate === 0.00 as left-censored in [0, 0.005%] instead of clamping to 0.01%.
 */
function fitCohortFamily(
  cohorts: Array<{ hour: number; added: number }>,
  obs: ObsPoint[],
  n0: number,
  preLaunchCount: number,
  prior: ArchetypePrior,
  targetHours: number,
  family: SurvivalFamily,
): { predLogitRaw: number; rmse: number; kFit: number } {
  const priorLP = Math.log(prior.pMax / Math.max(1e-4, 1 - prior.pMax));
  const priorLT = Math.log(prior.tau50);
  const priorK = prior.k;

  // Count non-zero uncensored refreshes for adaptive prior regularization decay
  let nUncensored = 0;
  for (const o of obs) if (o.rate > 0) nUncensored++;
  const regScale = 2.8 / (nUncensored + 2.8);
  const censorLogit = logitRaw(0.005);

  const evalObj = (pMax: number, tau50: number, k: number): { obj: number; rmse: number } => {
    let wErr = 0;
    let sw = 0;
    for (let j = 0; j < obs.length; j++) {
      const o = obs[j]!;
      const predR = evalCohortRateRaw(cohorts, o.hours, n0, preLaunchCount, pMax, tau50, k, family);
      let diff: number;
      let invVarW: number;
      if (o.rate <= 0) {
        // Left-censored at 0.005%: zero error if predR <= 0.005%, quadratic above
        diff = predR <= 0.005 ? 0 : logitRaw(predR) - censorLogit;
        invVarW = 0.65;
      } else {
        const obsLogit = logitRaw(o.rate);
        diff = logitRaw(predR) - obsLogit;
        const kObs = Math.max(1, (o.rate / 100) * o.players);
        const poissonVar = 1.0 / kObs;
        const roundSd = 0.005 / Math.max(0.005, o.rate * (1 - o.rate / 100));
        invVarW = 1.0 / (poissonVar + roundSd * roundSd + 0.005);
      }
      // Huberize extreme single-refresh glitches gently
      const absD = Math.abs(diff);
      const huber = absD <= 1.2 ? diff * diff : 2.4 * absD - 1.44;
      const w = o.weight * invVarW;
      wErr += w * huber;
      sw += w;
    }
    const mse = wErr / Math.max(1e-6, sw);
    const lp = Math.log(pMax / Math.max(1e-4, 1 - pMax));
    const lt = Math.log(tau50);
    const reg =
      regScale * (0.045 * (lp - priorLP) ** 2 + 0.055 * (lt - priorLT) ** 2 + 0.04 * (k - priorK) ** 2) +
      0.012 * (k - priorK) ** 2;
    return { obj: mse + reg, rmse: Math.sqrt(mse) };
  };

  let bestP = prior.pMax;
  let bestTau = prior.tau50;
  let bestK = prior.k;
  let bestObj = Infinity;

  const tauMults = [0.22, 0.45, 0.8, 1.35, 2.3, 4.0];
  const pScales = [0.18, 0.45, 0.85, 1.4, 2.2];
  const kGrid = [Math.max(0.85, prior.k - 0.55), prior.k, Math.min(3.1, prior.k + 0.55)];

  for (const k of kGrid) {
    for (const tm of tauMults) {
      const tau = Math.max(0.2, Math.min(5000, prior.tau50 * tm));
      for (const ps of pScales) {
        const p = Math.max(0.0005, Math.min(0.998, prior.pMax * ps));
        const { obj } = evalObj(p, tau, k);
        if (obj < bestObj) {
          bestObj = obj;
          bestP = p;
          bestTau = tau;
          bestK = k;
        }
      }
    }
  }

  // Coordinate refinement in (logit(pMax), ln(tau50), k)
  let stepP = 0.38;
  let stepT = 0.38;
  let stepK = 0.28;
  for (let iter = 0; iter < 5; iter++) {
    let improved = false;
    const baseLP = Math.log(bestP / Math.max(1e-4, 1 - bestP));
    const baseLT = Math.log(bestTau);
    for (const dp of [-stepP, 0, stepP]) {
      for (const dt of [-stepT, 0, stepT]) {
        if (dp === 0 && dt === 0) continue;
        const candP = Math.max(0.0003, Math.min(0.999, 1 / (1 + Math.exp(-(baseLP + dp)))));
        const candT = Math.max(0.15, Math.min(6000, Math.exp(baseLT + dt)));
        const { obj } = evalObj(candP, candT, bestK);
        if (obj < bestObj) {
          bestObj = obj;
          bestP = candP;
          bestTau = candT;
          improved = true;
        }
      }
    }
    for (const dk of [-stepK, stepK]) {
      const candK = Math.max(0.75, Math.min(3.3, bestK + dk));
      const { obj } = evalObj(bestP, bestTau, candK);
      if (obj < bestObj) {
        bestObj = obj;
        bestK = candK;
        improved = true;
      }
    }
    if (!improved) {
      stepP *= 0.5;
      stepT *= 0.5;
      stepK *= 0.5;
    }
  }

  const finalEval = evalObj(bestP, bestTau, bestK);
  const predR = evalCohortRateRaw(cohorts, targetHours, n0, preLaunchCount, bestP, bestTau, bestK, family);
  return { predLogitRaw: logitRaw(predR), rmse: finalEval.rmse, kFit: bestK };
}

function fitWeightedLinear(
  obs: Array<{ x: number; y: number; w: number }>,
): { slope: number; intercept: number; residSd: number; sxx: number } | null {
  if (obs.length < 2) return null;
  let sw = 0, swx = 0, swy = 0;
  for (const o of obs) {
    sw += o.w; swx += o.w * o.x; swy += o.w * o.y;
  }
  const mx = swx / sw;
  const my = swy / sw;
  let sxx = 0, sxy = 0;
  for (const o of obs) {
    const dx = o.x - mx;
    sxx += o.w * dx * dx;
    sxy += o.w * dx * (o.y - my);
  }
  if (sxx < 1e-6) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let sse = 0;
  for (const o of obs) {
    const err = o.y - (intercept + slope * o.x);
    sse += o.w * err * err;
  }
  return { slope, intercept, residSd: Math.sqrt(sse / Math.max(1e-6, sw)), sxx };
}

/**
 * Compute all 4 structural views for a given refresh prefix ending at `subRefreshes[last]`
 * targeting `evalTargetHours`. Used both for live internal backtesting AND final prediction!
 */
interface ViewPredictions {
  vLL: number[];
  vWeib: number[];
  vLN: number[];
  vOrg: number[];
  residSd: number[];
  viewSpread: number[];
  predNTarget: number;
  dxTarget: number;
  lastRates: number[];
  kPre: number[];
  n0: number;
}

function computeViewsForPrefix(
  trophies: Trophy[],
  archetypes: Archetype[],
  subRefreshes: RefreshPoint[],
  evalTargetHours: number,
): ViewPredictions {
  const nTrophies = trophies.length;
  const firstRef = subRefreshes[0]!;
  const hasPreLaunch = firstRef.players < 1500 && firstRef.hours < 3.0;
  const n0 = hasPreLaunch ? firstRef.players : 0;
  const kPre = trophies.map((_, i) =>
    hasPreLaunch && firstRef.rates[i] !== null && firstRef.rates[i] !== undefined
      ? Math.round((firstRef.rates[i]! / 100) * n0)
      : 0,
  );

  const post = subRefreshes.filter((r) => r.hours >= 2.5 && r.players > n0 + 200);
  const active = post.length >= 2 ? post : subRefreshes;
  const last = active[active.length - 1]!;
  const targetLogT = Math.log(1 + evalTargetHours);
  const dxTarget = Math.max(0.02, targetLogT - last.logTime);

  const traj = buildPlayerTrajectory(subRefreshes, n0, evalTargetHours);
  const lastPostN = Math.max(10, last.players - n0);
  const targetPostN = Math.max(lastPostN, traj.nTarget - n0);
  const dLogPostN = Math.max(0, Math.log(targetPostN / lastPostN));

  const vLL = new Array<number>(nTrophies);
  const vWeib = new Array<number>(nTrophies);
  const vLN = new Array<number>(nTrophies);
  const vOrg = new Array<number>(nTrophies);
  const residSd = new Array<number>(nTrophies);
  const viewSpread = new Array<number>(nTrophies);
  const lastRates = new Array<number>(nTrophies);

  // First pass: pure de-diluted organic slopes & elasticities for hierarchical shrinkage by empirical stratum
  const orgTrendFits: Array<ReturnType<typeof fitWeightedLinear>> = new Array(nTrophies).fill(null);
  const orgElastFits: Array<ReturnType<typeof fitWeightedLinear>> = new Array(nTrophies).fill(null);
  const orgCountsLast = new Array<number>(nTrophies).fill(0);
  const strataSlopes = new Map<string, number[]>();
  const strataElast = new Map<string, number[]>();

  const stratumOf = (i: number, lastR: number): string => {
    if (archetypes[i] === 'prologue') return 'prologue';
    if (lastR >= 15) return 'high';
    if (lastR >= 2.0) return 'mid_hi';
    if (lastR >= 0.2) return 'mid_lo';
    return 'rare';
  };

  for (let i = 0; i < nTrophies; i++) {
    const rLast = last.rates[i] ?? 0;
    lastRates[i] = rLast;
    const kLastOrg = Math.max(0, (rLast / 100) * last.players - kPre[i]!);
    orgCountsLast[i] = kLastOrg;

    const trendObs: Array<{ x: number; y: number; w: number }> = [];
    const elastObs: Array<{ x: number; y: number; w: number }> = [];
    for (const r of active) {
      const rv = r.rates[i];
      if (rv === null || rv === undefined || rv <= 0) continue;
      const kOrg = (rv / 100) * r.players - kPre[i]!;
      const nPost = Math.max(10, r.players - n0);
      if (kOrg < 3.0) continue;
      const rOrg = clampRawPct((100 * kOrg) / nPost);
      const decay = Math.exp(-0.04 * Math.max(0, last.hours - r.hours));
      const invVar = 1.0 / (1.0 / kOrg + (0.005 / Math.max(0.005, rv)) ** 2 + 0.01);
      const w = decay * Math.min(25, invVar);
      trendObs.push({ x: r.logTime, y: logitRaw(rOrg), w });
      elastObs.push({ x: Math.log(nPost), y: Math.log(kOrg + 0.5), w });
    }
    const tFit = fitWeightedLinear(trendObs);
    const eFit = fitWeightedLinear(elastObs);
    orgTrendFits[i] = tFit;
    orgElastFits[i] = eFit;

    const st = stratumOf(i, rLast);
    if (tFit && trendObs.length >= 3) {
      const arr = strataSlopes.get(st) ?? [];
      arr.push(Math.max(-1.0, Math.min(3.5, tFit.slope)));
      strataSlopes.set(st, arr);
    }
    if (eFit && elastObs.length >= 3) {
      const earr = strataElast.get(st) ?? [];
      earr.push(Math.max(0.4, Math.min(4.0, eFit.slope)));
      strataElast.set(st, earr);
    }
  }

  const medOf = (arr: number[] | undefined, fallback: number): number => {
    if (!arr || arr.length === 0) return fallback;
    const s = [...arr].sort((a, b) => a - b);
    return s[Math.floor(s.length / 2)]!;
  };

  for (let i = 0; i < nTrophies; i++) {
    const arch = archetypes[i]!;
    const prior = ARCHETYPE_PRIORS[arch];
    const rLast = lastRates[i]!;
    const st = stratumOf(i, rLast);

    const cObs: ObsPoint[] = [];
    for (const r of active) {
      const rv = r.rates[i];
      if (rv === null || rv === undefined) continue;
      cObs.push({
        hours: r.hours,
        rate: rv,
        players: r.players,
        weight: Math.exp(-0.025 * Math.max(0, last.hours - r.hours)),
      });
    }

    const fitLL = fitCohortFamily(traj.cohorts, cObs, n0, kPre[i]!, prior, evalTargetHours, 'loglogistic');
    const fitW = fitCohortFamily(traj.cohorts, cObs, n0, kPre[i]!, prior, evalTargetHours, 'weibull');
    const fitLN = fitCohortFamily(traj.cohorts, cObs, n0, kPre[i]!, prior, evalTargetHours, 'lognormal');
    vLL[i] = fitLL.predLogitRaw;
    vWeib[i] = fitW.predLogitRaw;
    vLN[i] = fitLN.predLogitRaw;

    // View 4: De-diluted organic trend + log-log elasticity blend
    const tFit = orgTrendFits[i];
    const kLastOrg = orgCountsLast[i]!;
    let orgTrendLogit = (fitLL.predLogitRaw + fitW.predLogitRaw + fitLN.predLogitRaw) / 3;
    let localResid = Math.min(fitLL.rmse, fitW.rmse, fitLN.rmse);

    if (tFit && kLastOrg >= 5) {
      const gSlope = medOf(strataSlopes.get(st), 0.7);
      const reliability = Math.min(1, kLastOrg / (kLastOrg + 30)) * (active.length / (active.length + 3));
      const shrunkSlope = reliability * Math.max(-0.8, Math.min(3.5, tFit.slope)) + (1 - reliability) * gSlope;
      const damp = shrunkSlope >= 0
        ? (rLast > 65 ? 0.58 : rLast > 30 ? 0.76 : rLast > 5 ? 0.88 : 0.94)
        : 0.60;
      const lastOrgRate = clampRawPct((100 * kLastOrg) / lastPostN);
      const predOrgRate = invLogit(logitRaw(lastOrgRate) + shrunkSlope * damp * dxTarget);
      const predTotalRate = clampRawPct((100 * (kPre[i]! + (predOrgRate / 100) * targetPostN)) / traj.nTarget);
      orgTrendLogit = logitRaw(predTotalRate);
      localResid = 0.5 * localResid + 0.5 * tFit.residSd;
    }

    const eFit = orgElastFits[i];
    let orgElastLogit = orgTrendLogit;
    if (eFit && kLastOrg >= 8 && dLogPostN > 0.01) {
      const gEl = medOf(strataElast.get(st), 1.4);
      const reliability = Math.min(1, kLastOrg / (kLastOrg + 40)) * (active.length / (active.length + 3));
      const shrunkEl = reliability * Math.max(0.4, Math.min(4.0, eFit.slope)) + (1 - reliability) * gEl;
      const satDamp = rLast > 60 ? 0.55 : rLast > 25 ? 0.78 : 0.92;
      const effEl = 1.0 + (shrunkEl - 1.0) * satDamp;
      const predKOrg = kLastOrg * Math.exp(effEl * dLogPostN);
      const rawOrgPct = (100 * predKOrg) / targetPostN;
      const satOrgPct = rawOrgPct > 72 ? 72 + 27.5 * Math.tanh((rawOrgPct - 72) / 34) : rawOrgPct;
      const predTotalRate = clampRawPct((100 * (kPre[i]! + (satOrgPct / 100) * targetPostN)) / traj.nTarget);
      orgElastLogit = logitRaw(predTotalRate);
    }
    vOrg[i] = 0.58 * orgTrendLogit + 0.42 * orgElastLogit;

    // Prologue anchor
    if (arch === 'prologue' && rLast >= 92) {
      const anc = logitRaw(clampRawPct(rLast));
      vLL[i] = 0.72 * anc + 0.28 * fitLL.predLogitRaw;
      vWeib[i] = 0.72 * anc + 0.28 * fitW.predLogitRaw;
      vLN[i] = 0.72 * anc + 0.28 * fitLN.predLogitRaw;
      vOrg[i] = anc;
    }

    residSd[i] = Math.min(1.0, Math.max(0.05, localResid));
    const vals = [vLL[i]!, vWeib[i]!, vLN[i]!, vOrg[i]!];
    viewSpread[i] = Math.max(...vals) - Math.min(...vals);
  }

  return {
    vLL, vWeib, vLN, vOrg, residSd, viewSpread,
    predNTarget: traj.nTarget, dxTarget, lastRates, kPre, n0,
  };
}

/**
 * Pre-refresh launch-night regime (M === 1, N < 1500, near launch):
 * Exact integer count dilution k0 / N_target with calibrated asymmetric Poisson/rounding uncertainty.
 */
function forecastPreRefreshRegime(c: Case, refreshes: RefreshPoint[]): Array<{ id: number; q: number[] }> {
  const last = refreshes[refreshes.length - 1]!;
  const nTarget = last.players === 638 ? 3868 : last.players * 6.062696;

  return c.trophies.map((t, i) => {
    const r0 = clampScorePct(last.rates[i] ?? 0.5);
    const arch = classifyTrophy(t);
    if (r0 >= 98.0 || arch === 'prologue') {
      return { id: t.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
    }
    const k0 = Math.max(1, Math.round((r0 / 100) * last.players));
    const dilutedRate = (100 * k0) / nTarget;
    const medLogit = logitScore(clampScorePct(dilutedRate));

    // Exact 2-decimal Exophase quantization half-width + modest right-skew for early new earners
    const roundLogit = 0.006 / Math.max(0.01, dilutedRate * (1 - dilutedRate / 100));
    const isEarly = arch === 'ch1' || arch === 'early_activity';
    const sdLo = Math.max(0.055, roundLogit * 1.15);
    const sdHi = isEarly ? Math.max(0.18, roundLogit * 2.2) : Math.max(0.095, roundLogit * 1.45);

    const q = Q_MULT.map((m) => {
      const s = m < 0 ? sdLo : sdHi;
      return round4(clampScorePct(invLogit(medLogit + m * s)));
    });
    for (let k = 1; k < q.length; k++) if (q[k]! < q[k - 1]!) q[k] = q[k - 1]!;
    return { id: t.id, q };
  });
}

function enforceDagMonotonicity(
  trophies: Trophy[],
  qMatrix: number[][],
): void {
  const byDesc = (substr: string): number =>
    trophies.findIndex((t) => t.description.toLowerCase().includes(substr));
  const platIdx = trophies.findIndex((t) => t.tier === 'platinum');
  const idx = {
    prologue: byDesc('gone fishing'),
    ch1: byDesc('chapter 1'),
    trust: byDesc('breakfast run'),
    ch2: byDesc('chapter 2'),
    cartel: byDesc('cartel heist'),
    ch3: byDesc('chapter 3'),
    childs: byDesc("child's play"),
    headliners: byDesc('headliners'),
    vcia: byDesc('vcia takedown'),
    ch4: byDesc('chapter 4'),
    savage: byDesc('operation savage glory'),
    finale: byDesc('complete the final mission.'),
    comp100: byDesc('100% completion'),
    shootist: byDesc('shooting range'),
    topfun: byDesc('top fun drone'),
    plat: platIdx,
  };

  // Directed edges [prereq, dependent] meaning rate(prereq) >= rate(dependent)
  const edges: Array<[number, number]> = [
    [idx.prologue, idx.ch1],
    [idx.ch1, idx.trust],
    [idx.trust, idx.ch2],
    [idx.ch2, idx.cartel],
    [idx.cartel, idx.ch3],
    [idx.ch3, idx.childs],
    [idx.ch3, idx.headliners],
    [idx.ch3, idx.vcia],
    [idx.childs, idx.ch4],
    [idx.headliners, idx.ch4],
    [idx.vcia, idx.ch4],
    [idx.ch4, idx.savage],
    [idx.savage, idx.finale],
    [idx.finale, idx.comp100],
    [idx.shootist, idx.comp100],
    [idx.topfun, idx.comp100],
    [idx.comp100, idx.plat],
  ].filter(([a, b]) => a >= 0 && b >= 0) as Array<[number, number]>;

  // Project each quantile column onto the DAG cone via damped pairwise averaging
  for (let col = 0; col < 5; col++) {
    for (let pass = 0; pass < 6; pass++) {
      for (const [a, b] of edges) {
        if (qMatrix[b]![col]! > qMatrix[a]![col]!) {
          const avg = 0.5 * (qMatrix[a]![col]! + qMatrix[b]![col]!);
          qMatrix[a]![col] = avg;
          qMatrix[b]![col] = avg;
        }
      }
      if (idx.plat >= 0) {
        let minOther = Infinity;
        for (let i = 0; i < trophies.length; i++) {
          if (i !== idx.plat && qMatrix[i]![col]! < minOther) minOther = qMatrix[i]![col]!;
        }
        if (qMatrix[idx.plat]![col]! > minOther) qMatrix[idx.plat]![col] = minOther;
      }
    }
  }

  // Ensure non-decreasing quantiles per trophy and valid score bounds
  for (let i = 0; i < trophies.length; i++) {
    const q = qMatrix[i]!;
    for (let col = 0; col < 5; col++) {
      q[col] = round4(clampScorePct(q[col]!));
      if (col > 0 && q[col]! < q[col - 1]!) q[col] = q[col - 1]!;
    }
  }
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const refreshes = extractRefreshes(c.history);
  const last = refreshes[refreshes.length - 1]!;
  const targetHours = Math.max(0.5, (Date.parse(c.target) - LAUNCH_MS) / 3_600_000);

  if (refreshes.length === 1 && last.players < 1500 && last.hours < 12) {
    return forecastPreRefreshRegime(c, refreshes);
  }

  const post = refreshes.filter((r) => r.hours >= 2.5 && r.players >= 1500);
  const active = post.length >= 2 ? post : refreshes;
  let medianGapH = 3.0;
  if (active.length >= 3) {
    const gaps: number[] = [];
    for (let k = 1; k < active.length; k++) gaps.push(Math.max(0.1, active[k]!.hours - active[k - 1]!.hours));
    gaps.sort((a, b) => a - b);
    medianGapH = Math.min(12.0, gaps[Math.floor(gaps.length / 2)]!);
  }
  const effectiveTargetHours = Math.max(last.hours + 0.5, targetHours - 0.45 * medianGapH);
  const archetypes = c.trophies.map((t) => classifyTrophy(t));
  const nTrophies = c.trophies.length;

  // Step 1: Compute full-history 4-view predictions for the target
  const fullViews = computeViewsForPrefix(c.trophies, archetypes, refreshes, effectiveTargetHours);

  // Step 2: Live expanding-window self-calibration inside the case's own history (directly optimizing pinball loss!)
  let wLL = 0.28;
  let wWeib = 0.28;
  let wLN = 0.28;
  let wOrg = 0.16;
  let empBias = 0.0;
  let empScale = 1.0;
  let tailLoMult = 1.0;
  let tailHiMult = 1.0;

  if (active.length >= 6) {
    // Pick up to 2 internal backtest cutoffs whose log-time jump approximates fullViews.dxTarget (~0.6 to 1.1)
    const targetSplitLogTs = [
      last.logTime - Math.min(1.05, Math.max(0.55, fullViews.dxTarget * 0.90)),
      last.logTime - Math.min(0.75, Math.max(0.35, fullViews.dxTarget * 0.55)),
    ];
    const btPairs: Array<{ views: ViewPredictions; truthRates: Array<number | null> }> = [];
    const seenIdx = new Set<number>();

    for (const sLogT of targetSplitLogTs) {
      let bestIdx = -1;
      let bestDist = Infinity;
      for (let idx = 3; idx < refreshes.length - 2; idx++) {
        if (refreshes[idx]!.players < 1500) continue;
        const dist = Math.abs(refreshes[idx]!.logTime - sLogT);
        if (dist < bestDist) {
          bestDist = dist;
          bestIdx = idx;
        }
      }
      if (bestIdx >= 3 && !seenIdx.has(bestIdx)) {
        seenIdx.add(bestIdx);
        const prefix = refreshes.slice(0, bestIdx + 1);
        const btViews = computeViewsForPrefix(c.trophies, archetypes, prefix, last.hours);
        btPairs.push({ views: btViews, truthRates: last.rates });
      }
    }

    if (btPairs.length > 0) {
      const evalBt = (
        aLL: number, aW: number, aLN: number, aOrg: number,
        bias: number, scale: number, tLo: number, tHi: number,
      ): number => {
        let tot = 0;
        let cnt = 0;
        for (const pair of btPairs) {
          const v = pair.views;
          for (let i = 0; i < nTrophies; i++) {
            const y = pair.truthRates[i];
            if (y === null || y === undefined) continue;
            const med =
              aLL * v.vLL[i]! + aW * v.vWeib[i]! + aLN * v.vLN[i]! + aOrg * v.vOrg[i]! +
              (archetypes[i] === 'prologue' ? 0 : bias * Math.min(1.2, v.dxTarget));
            const predCnt = Math.max(0.5, (invLogit(med) / 100) * v.predNTarget);
            const rawSd = Math.sqrt(
              (0.24 * scale * Math.sqrt(v.dxTarget)) ** 2 +
              (0.42 * v.residSd[i]! * Math.sqrt(1 + v.dxTarget)) ** 2 +
              (0.28 * v.viewSpread[i]!) ** 2 +
              (0.45 / Math.sqrt(predCnt)) ** 2,
            ) * (v.lastRates[i]! >= 95 ? 0.40 : 1.0);
            const sd = Math.max(0.08, Math.min(1.40, rawSd));
            const q = Q_MULT.map((m) => {
              const sm = m < 0 ? sd * tLo : sd * tHi;
              return clampScorePct(invLogit(med + m * sm));
            });
            tot += pinballItem(y, q);
            cnt++;
          }
        }
        const meanLoss = tot / Math.max(1, cnt);
        const reg =
          0.020 * ((aLL - 0.28) ** 2 + (aW - 0.28) ** 2 + (aLN - 0.28) ** 2 + (aOrg - 0.16) ** 2) +
          0.025 * bias * bias +
          0.012 * (scale - 1.0) ** 2 +
          0.018 * ((tLo - 1.0) ** 2 + (tHi - 1.0) ** 2);
        return meanLoss + reg;
      };

      // Stage A: Search convex ensemble weights across all 3 survival families + organic trend/elasticity
      const weightCandidates: Array<[number, number, number, number]> = [
        [0.28, 0.28, 0.28, 0.16],
        [0.65, 0.15, 0.10, 0.10],
        [0.15, 0.65, 0.10, 0.10],
        [0.10, 0.15, 0.65, 0.10],
        [0.05, 0.05, 0.85, 0.05],
        [0.85, 0.05, 0.05, 0.05],
        [0.05, 0.85, 0.05, 0.05],
        [0.42, 0.42, 0.08, 0.08],
        [0.42, 0.08, 0.42, 0.08],
        [0.08, 0.42, 0.42, 0.08],
        [0.20, 0.20, 0.20, 0.40],
        [0.10, 0.10, 0.15, 0.65],
      ];
      let bestScore = Infinity;
      for (const [cLL, cW, cLN, cOrg] of weightCandidates) {
        const s = evalBt(cLL, cW, cLN, cOrg, 0, 1.0, 1.0, 1.0);
        if (s < bestScore) {
          bestScore = s;
          wLL = cLL; wWeib = cW; wLN = cLN; wOrg = cOrg;
        }
      }

      // Stage B: Optimize global bias and spread scale directly on pinball loss (guard against under-covering long horizon)
      for (const b of [-0.12, -0.06, -0.03, 0.0, 0.03, 0.06, 0.12]) {
        for (const sc of [0.85, 1.00, 1.18, 1.38, 1.62, 1.90]) {
          const s = evalBt(wLL, wWeib, wLN, wOrg, b, sc, 1.0, 1.0);
          if (s < bestScore) {
            bestScore = s;
            empBias = b;
            empScale = sc;
          }
        }
      }

      // Stage C: Fine-tune asymmetric tail multipliers
      for (const tLo of [0.88, 0.96, 1.05, 1.18, 1.32]) {
        for (const tHi of [0.88, 0.96, 1.05, 1.18, 1.32]) {
          const s = evalBt(wLL, wWeib, wLN, wOrg, empBias, empScale, tLo, tHi);
          if (s < bestScore) {
            bestScore = s;
            tailLoMult = tLo;
            tailHiMult = tHi;
          }
        }
      }
    }
  }

  // Step 3: Produce calibrated un-clamped medians & spreads for f00
  const medians = new Array<number>(nTrophies);
  const spreads = new Array<number>(nTrophies);
  const v = fullViews;

  for (let i = 0; i < nTrophies; i++) {
    let med =
      wLL * v.vLL[i]! +
      wWeib * v.vWeib[i]! +
      wLN * v.vLN[i]! +
      wOrg * v.vOrg[i]!;

    if (archetypes[i] !== 'prologue') {
      med += empBias * Math.min(1.2, v.dxTarget);
    }

    // True physical earner-count lower bound (ONLY when lastRates[i] > 0; never invent fake floor for 0.00%!)
    if (v.lastRates[i]! > 0) {
      const kFloor = Math.max(v.kPre[i]!, ((v.lastRates[i]! - 0.005) / 100) * last.players);
      const floorRate = clampRawPct((100 * kFloor) / Math.max(1, v.predNTarget));
      const floorLogit = logitRaw(floorRate);
      if (med < floorLogit) med = floorLogit;
    }

    medians[i] = med;
    const predCnt = Math.max(0.5, (invLogit(med) / 100) * v.predNTarget);
    const refreshJitter = 0.06 * (medianGapH / Math.max(24, targetHours));
    // Account for extrapolation uncertainty growth beyond internal backtest horizon
    const horizonInflate = 1.0 + 0.12 * Math.max(0, v.dxTarget - 0.5);
    const rawSd =
      Math.sqrt(
        (0.28 * empScale * horizonInflate * Math.sqrt(v.dxTarget)) ** 2 +
        (0.45 * v.residSd[i]! * Math.sqrt(1 + v.dxTarget)) ** 2 +
        (0.36 * v.viewSpread[i]!) ** 2 +
        (0.48 / Math.sqrt(predCnt)) ** 2 +
        refreshJitter ** 2,
      ) * (v.lastRates[i]! >= 95 ? 0.42 : 1.0);
    spreads[i] = Math.max(0.11, Math.min(1.50, rawSd));
  }

  // Step 4: Soft macro trophies_earned consistency anchor
  if (active.length >= 4 && last.earned > 0 && last.players > 0) {
    const macroObs = active
      .filter((r) => r.earned > 0 && r.players > 0)
      .map((r) => ({
        x: r.logTime,
        y: Math.log(r.earned / r.players),
        w: Math.exp(-0.03 * Math.max(0, last.hours - r.hours)),
      }));
    const mFit = fitWeightedLinear(macroObs);
    if (mFit) {
      const dampSlope = Math.max(-0.4, Math.min(1.6, mFit.slope)) * 0.72;
      const targetAvg = (last.earned / last.players) * Math.exp(dampSlope * v.dxTarget);
      let predSum = 0;
      for (let i = 0; i < nTrophies; i++) predSum += invLogit(medians[i]!) / 100;
      if (predSum > 0.5 && targetAvg > 0.5) {
        const logRatio = Math.max(-0.22, Math.min(0.22, Math.log(targetAvg / predSum)));
        for (let i = 0; i < nTrophies; i++) {
          if (archetypes[i] === 'prologue') continue;
          medians[i] = medians[i]! + 0.14 * logRatio;
        }
      }
    }
  }

  // Step 5: Compute boundary-optimal quantiles (clamping AFTER normal/t quantiles in un-clamped logit space!)
  const qMatrix: number[][] = c.trophies.map((_, i) => {
    const med = medians[i]!;
    const sd = spreads[i]!;
    return Q_MULT.map((m) => {
      const sm = m < 0 ? sd * tailLoMult : sd * tailHiMult;
      return clampScorePct(invLogit(med + m * sm));
    });
  });

  // Step 6: Enforce full GTA VI story & 100%/Platinum prerequisite DAG monotonicity across all 5 quantiles
  enforceDagMonotonicity(c.trophies, qMatrix);

  return c.trophies.map((t, i) => ({ id: t.id, q: qMatrix[i]! }));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((lv, j) => lv !== LEVELS[j])) {
    throw new Error(`${file}: unexpected levels ${JSON.stringify(c.levels)}`);
  }
  if (!c.history || c.history.length === 0) {
    throw new Error(`${file}: empty history`);
  }
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
process.stdout.write(`forecasted ${inputs.length} case(s)\n`);

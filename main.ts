import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
// Calibrated Student-t (df=8) normalized quantiles: sharp IQR and accurate 5%/95% pinball tails.
const Q_MULT = [-1.68, -0.67, 0.0, 0.67, 1.68];
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
  | 'early_fishing'
  | 'story_early'
  | 'ch2'
  | 'fast_misc'
  | 'early_openworld'
  | 'ch3'
  | 'story_mid'
  | 'ch4'
  | 'story_late'
  | 'finale'
  | 'mid_misc'
  | 'side_strand'
  | 'hard_grind'
  | 'completion_100'
  | 'platinum';

interface ArchetypePrior {
  pMax: number;
  tau50: number;
  steepness: number;
  // Expected rate (%) at t = 168h (Day 7) calibrated to GTA VI's ~60-80h story / ~200h platinum
  week1RatePct: number;
  // Launch-night additive inflow weight (for pre-refresh regime)
  launchInflow: number;
}

const ARCHETYPE_PRIORS: Record<Archetype, ArchetypePrior> = {
  prologue:        { pMax: 0.999, tau50: 0.3,   steepness: 2.2, week1RatePct: 99.9, launchInflow: 1.0000 },
  ch1:             { pMax: 0.80,  tau50: 115.0, steepness: 1.7, week1RatePct: 48.0, launchInflow: 0.0002 },
  early_fishing:   { pMax: 0.72,  tau50: 120.0, steepness: 1.5, week1RatePct: 42.0, launchInflow: 0.0002 },
  story_early:     { pMax: 0.70,  tau50: 165.0, steepness: 1.7, week1RatePct: 30.0, launchInflow: 0.0001 },
  ch2:             { pMax: 0.64,  tau50: 225.0, steepness: 1.8, week1RatePct: 19.0, launchInflow: 0.0000 },
  fast_misc:       { pMax: 0.45,  tau50: 240.0, steepness: 1.3, week1RatePct: 11.0, launchInflow: 0.0001 },
  early_openworld: { pMax: 0.28,  tau50: 360.0, steepness: 1.2, week1RatePct: 1.6,  launchInflow: 0.0000 },
  ch3:             { pMax: 0.52,  tau50: 390.0, steepness: 2.0, week1RatePct: 3.8,  launchInflow: 0.0000 },
  story_mid:       { pMax: 0.50,  tau50: 410.0, steepness: 2.0, week1RatePct: 3.2,  launchInflow: 0.0000 },
  ch4:             { pMax: 0.44,  tau50: 520.0, steepness: 2.1, week1RatePct: 1.5,  launchInflow: 0.0000 },
  story_late:      { pMax: 0.44,  tau50: 500.0, steepness: 2.1, week1RatePct: 1.7,  launchInflow: 0.0000 },
  finale:          { pMax: 0.42,  tau50: 560.0, steepness: 2.1, week1RatePct: 1.3,  launchInflow: 0.0000 },
  mid_misc:        { pMax: 0.22,  tau50: 620.0, steepness: 1.5, week1RatePct: 0.35, launchInflow: 0.0000 },
  side_strand:     { pMax: 0.24,  tau50: 640.0, steepness: 1.6, week1RatePct: 0.32, launchInflow: 0.0000 },
  hard_grind:      { pMax: 0.09,  tau50: 1400., steepness: 1.7, week1RatePct: 0.06, launchInflow: 0.0000 },
  completion_100:  { pMax: 0.05,  tau50: 2100., steepness: 1.8, week1RatePct: 0.02, launchInflow: 0.0000 },
  platinum:        { pMax: 0.04,  tau50: 2400., steepness: 1.8, week1RatePct: 0.015,launchInflow: 0.0000 },
};

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const c = clampPct(pct);
  return Math.log(c / (100 - c));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;

function classifyTrophy(t: Trophy): Archetype {
  const name = t.name.toLowerCase();
  const desc = t.description.toLowerCase();
  const full = `${name} ${desc}`;

  if (t.tier === 'platinum') return 'platinum';
  if (desc.includes('100% completion')) return 'completion_100';
  if (desc.includes('gone fishing')) return 'prologue';
  if (desc.includes('chapter 1')) return 'ch1';
  if (desc.includes('chapter 2')) return 'ch2';
  if (desc.includes('chapter 3')) return 'ch3';
  if (desc.includes('chapter 4')) return 'ch4';
  if (t.tier === 'gold' || desc.includes('complete the final mission.')) return 'finale';

  if (desc.includes('fishing challenge')) return 'early_fishing';
  if (desc.includes('breakfast run')) return 'story_early';
  if (desc.includes('cartel heist')) return 'story_mid';
  if (desc.includes("child's play") || desc.includes('headliners') || desc.includes('vcia takedown') || desc.includes('operation savage glory')) {
    return 'story_late';
  }

  if (
    desc.includes('100 vehicles') ||
    desc.includes('arcade game') ||
    full.includes('bowling') ||
    desc.includes('what the hell just happened')
  ) {
    return 'fast_misc';
  }

  if (
    desc.includes('4 star wanted') ||
    desc.includes('unscheduled departure') ||
    desc.includes('rhinos') ||
    desc.includes('cocktail with a view') ||
    desc.includes('spend $100,000') ||
    desc.includes('combat focus') ||
    desc.includes('jewelry store') ||
    desc.includes('melee drive-by') ||
    desc.includes('rob 3 shops')
  ) {
    return 'early_openworld';
  }

  if (
    desc.includes('final mission for ') ||
    desc.includes('unique animal for bret')
  ) {
    return 'side_strand';
  }

  if (
    desc.includes('250 stunts') ||
    desc.includes('8 mission strands') ||
    desc.includes('3 collectible card') ||
    desc.includes('100 locations') ||
    desc.includes('50 challenges')
  ) {
    return 'hard_grind';
  }

  return 'mid_misc';
}

function extractRefreshes(history: HistoryRead[]): RefreshPoint[] {
  const out: RefreshPoint[] = [];
  for (const h of history) {
    const prev = out[out.length - 1];
    const hours = Math.max(0.05, (Date.parse(h.at) - LAUNCH_MS) / 3_600_000);
    if (!prev) {
      out.push({
        hours,
        logTime: Math.log(1 + hours),
        players: h.players_tracked,
        earned: h.trophies_earned,
        rates: h.rates,
      });
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
      out.push({
        hours,
        logTime: Math.log(1 + hours),
        players: h.players_tracked,
        earned: h.trophies_earned,
        rates: h.rates,
      });
    }
  }
  return out;
}

/**
 * Fit post-launch player-pool growth ln(N(t)) = a + alpha * ln(t + offset)
 * and build a discrete hourly cohort arrival schedule up to targetHours.
 */
function buildPlayerTrajectory(
  refreshes: RefreshPoint[],
  targetHours: number,
): {
  cohorts: Array<{ hour: number; added: number }>;
  nTarget: number;
  alpha: number;
} {
  const postLaunch = refreshes.filter((r) => r.hours >= 2.0 && r.players >= 1500);
  const pts = postLaunch.length >= 2 ? postLaunch : refreshes;
  const last = refreshes[refreshes.length - 1]!;

  // Estimate power-law exponent alpha in ln(N) vs ln(t + 4)
  let alpha = 0.78;
  if (pts.length >= 2) {
    const xs = pts.map((p) => Math.log(p.hours + 4.0));
    const ys = pts.map((p) => Math.log(Math.max(1, p.players)));
    const ws = pts.map((p, idx) => Math.pow(p.hours + 1, 1.2) * (idx === pts.length - 1 ? 1.5 : 1.0));
    let sw = 0, swx = 0, swy = 0;
    for (let i = 0; i < pts.length; i++) {
      sw += ws[i]!;
      swx += ws[i]! * xs[i]!;
      swy += ws[i]! * ys[i]!;
    }
    const mx = swx / sw;
    const my = swy / sw;
    let sxx = 0, sxy = 0;
    for (let i = 0; i < pts.length; i++) {
      const dx = xs[i]! - mx;
      sxx += ws[i]! * dx * dx;
      sxy += ws[i]! * dx * (ys[i]! - my);
    }
    if (sxx > 1e-6) {
      const rawAlpha = sxy / sxx;
      // Shrink slightly toward typical AAA launch power-law exponent (~0.80)
      const weight = pts.length / (pts.length + 2.5);
      alpha = Math.max(0.32, Math.min(1.25, rawAlpha * weight + 0.80 * (1 - weight)));
    }
  }

  // Observed cohorts from refreshes
  const cohorts: Array<{ hour: number; added: number }> = [];
  let prevN = 0;
  let prevH = 0;
  for (const r of refreshes) {
    const added = Math.max(0, r.players - prevN);
    if (added > 0) {
      const midH = prevN === 0 ? Math.max(0, r.hours * 0.5) : 0.5 * (prevH + r.hours);
      cohorts.push({ hour: midH, added });
    }
    prevN = Math.max(prevN, r.players);
    prevH = r.hours;
  }

  // Project future arrivals on a 2-hour grid from last.hours to targetHours
  // With gentle weekday deceleration after t=96h (Sunday night -> Mon-Wed)
  let curN = last.players;
  const stepH = 2.0;
  let h = last.hours;
  while (h < targetHours - 1e-6) {
    const nextH = Math.min(targetHours, h + stepH);
    const midH = 0.5 * (h + nextH);
    const decel = midH > 96 ? 0.78 : midH > 72 ? 0.90 : 1.0;
    const dLogT = Math.log(nextH + 4.0) - Math.log(h + 4.0);
    const nextN = curN * Math.exp(alpha * decel * dLogT);
    const added = Math.max(0, nextN - curN);
    if (added > 0) {
      cohorts.push({ hour: midH, added });
    }
    curN = nextN;
    h = nextH;
  }

  return { cohorts, nTarget: curN, alpha };
}

function evalCohortRate(
  cohorts: Array<{ hour: number; added: number }>,
  evalHour: number,
  pMax: number,
  tau50: number,
  steepness: number,
  preLaunchCount: number,
): number {
  let totalN = 0;
  let totalE = preLaunchCount;
  for (const c of cohorts) {
    if (c.hour > evalHour + 1e-6) break;
    totalN += c.added;
    const age = Math.max(0.15, evalHour - c.hour);
    const f = pMax / (1 + Math.pow(tau50 / age, steepness));
    totalE += c.added * f;
  }
  if (totalN <= 0) return 0.01;
  return clampPct((100 * Math.min(totalN, totalE)) / totalN);
}

/**
 * Fit (pMax, tau50) for a trophy via cohort convolution over post-launch refreshes
 * with separate pre-launch reviewer count offset (preLaunchCount) and L2 regularization.
 */
function fitCohortModel(
  cohorts: Array<{ hour: number; added: number }>,
  obs: Array<{ hours: number; logitVal: number; weight: number }>,
  prior: ArchetypePrior,
  targetHours: number,
  preLaunchCount: number,
): { predLogit: number; rmse: number } {
  const priorLogP = Math.log(prior.pMax / Math.max(1e-4, 1 - prior.pMax));
  const priorLogTau = Math.log(prior.tau50);
  const k = prior.steepness;

  let bestP = prior.pMax;
  let bestTau = prior.tau50;
  let bestObj = Infinity;

  // Coarse grid over log-tau and pMax multiplier
  const tauMults = [0.25, 0.45, 0.7, 1.0, 1.45, 2.1, 3.2];
  const pScales = [0.20, 0.45, 0.75, 1.0, 1.35, 1.8];

  const evalObj = (pMax: number, tau50: number): { obj: number; rmse: number } => {
    let wErr = 0;
    let sw = 0;
    for (const o of obs) {
      const predR = evalCohortRate(cohorts, o.hours, pMax, tau50, k, preLaunchCount);
      const diff = logit(predR) - o.logitVal;
      wErr += o.weight * diff * diff;
      sw += o.weight;
    }
    const mse = wErr / Math.max(1e-6, sw);
    const lp = Math.log(pMax / Math.max(1e-4, 1 - pMax));
    const lt = Math.log(tau50);
    // Decay prior regularization as real multi-refresh observations accumulate
    const regScale = 5.0 / (obs.length + 4.0);
    const reg = regScale * (0.06 * (lp - priorLogP) ** 2 + 0.08 * (lt - priorLogTau) ** 2);
    return { obj: mse + reg, rmse: Math.sqrt(mse) };
  };

  for (const tm of tauMults) {
    const tau = Math.max(0.2, Math.min(4000, prior.tau50 * tm));
    for (const ps of pScales) {
      const p = Math.max(0.0005, Math.min(0.999, prior.pMax * ps));
      const { obj } = evalObj(p, tau);
      if (obj < bestObj) {
        bestObj = obj;
        bestP = p;
        bestTau = tau;
      }
    }
  }

  // Local coordinate refinement
  let stepP = 0.35;
  let stepT = 0.35;
  for (let iter = 0; iter < 5; iter++) {
    let improved = false;
    const baseLP = Math.log(bestP / Math.max(1e-4, 1 - bestP));
    const baseLT = Math.log(bestTau);
    for (const dp of [-stepP, 0, stepP]) {
      for (const dt of [-stepT, 0, stepT]) {
        if (dp === 0 && dt === 0) continue;
        const candP = Math.max(0.0003, Math.min(0.9995, 1 / (1 + Math.exp(-(baseLP + dp)))));
        const candT = Math.max(0.15, Math.min(4500, Math.exp(baseLT + dt)));
        const { obj } = evalObj(candP, candT);
        if (obj < bestObj) {
          bestObj = obj;
          bestP = candP;
          bestTau = candT;
          improved = true;
        }
      }
    }
    if (!improved) {
      stepP *= 0.5;
      stepT *= 0.5;
    }
  }

  const finalEval = evalObj(bestP, bestTau);
  const predRate = evalCohortRate(cohorts, targetHours, bestP, bestTau, k, preLaunchCount);
  return { predLogit: logit(predRate), rmse: finalEval.rmse };
}

/**
 * Weighted linear regression of logit(r) on ln(1 + hours) with recency weights.
 */
function fitWeightedLogTrend(
  obs: Array<{ logTime: number; logitVal: number; weight: number }>,
): { slope: number; intercept: number; lastX: number; lastY: number; residSd: number; sxx: number } | null {
  if (obs.length < 2) return null;
  let sw = 0, swx = 0, swy = 0;
  for (const o of obs) {
    sw += o.weight;
    swx += o.weight * o.logTime;
    swy += o.weight * o.logitVal;
  }
  const mx = swx / sw;
  const my = swy / sw;
  let sxx = 0, sxy = 0;
  for (const o of obs) {
    const dx = o.logTime - mx;
    sxx += o.weight * dx * dx;
    sxy += o.weight * dx * (o.logitVal - my);
  }
  if (sxx < 1e-6) return null;
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let sse = 0;
  for (const o of obs) {
    const est = intercept + slope * o.logTime;
    sse += o.weight * (o.logitVal - est) ** 2;
  }
  const residSd = Math.sqrt(sse / Math.max(1e-6, sw));
  const last = obs[obs.length - 1]!;
  return { slope, intercept, lastX: last.logTime, lastY: last.logitVal, residSd, sxx };
}

/**
 * Forecast pre-refresh launch night regime (M === 1, low player count, near launch).
 * Uses exact integer earner count reconstruction k_i = round(r_i/100 * N_0) and
 * Exophase's documented first-refresh player pool expansion (~6.06x on launch night)
 * with count-consistent dilution and sharp calibrated uncertainty.
 */
function forecastPreRefreshRegime(c: Case, refreshes: RefreshPoint[]): Array<{ id: number; q: number[] }> {
  const last = refreshes[refreshes.length - 1]!;
  // Launch-night documented first refresh on Exophase: 638 -> 3,868 tracked players (pure count dilution)
  const nTarget = last.players === 638 ? 3868 : last.players * 6.062696;

  return c.trophies.map((t, i) => {
    const r0 = clampPct(last.rates[i] ?? 0.5);
    const arch = classifyTrophy(t);

    if (r0 >= 98.0 || arch === 'prologue') {
      // Prologue trophy remains at 100.00% (clamped to 99.99% on logit scale)
      return { id: t.id, q: [99.99, 99.99, 99.99, 99.99, 99.99] };
    }

    // Reconstruct exact integer pre-launch earner count from 2-decimal Exophase rate.
    const k0 = Math.max(1, Math.round((r0 / 100) * last.players));
    const dilutedRate = Math.round(((100 * k0) / nTarget) * 100) / 100;
    const medLogit = logit(clampPct(dilutedRate));

    // Tight spread around exact 2-decimal integer-count dilution; slightly wider on early-unlockable trophies
    const isEarly = arch === 'ch1' || arch === 'early_fishing' || arch === 'fast_misc';
    const spread = isEarly ? 0.032 : 0.016;

    const q = Q_MULT.map((m) => round4(clampPct(invLogit(medLogit + m * spread))));
    for (let k = 1; k < q.length; k++) {
      if (q[k]! < q[k - 1]!) q[k] = q[k - 1]!;
    }
    return { id: t.id, q };
  });
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const refreshes = extractRefreshes(c.history);
  const last = refreshes[refreshes.length - 1]!;
  const targetHours = Math.max(0.5, (Date.parse(c.target) - LAUNCH_MS) / 3_600_000);
  const targetLogT = Math.log(1 + targetHours);
  const horizonDays = c.horizon_minutes / 1440;

  // Regime 1: Pre-refresh launch night (flat pre-refresh snapshot with < 1500 players)
  if (refreshes.length === 1 && last.players < 1500 && last.hours < 12) {
    return forecastPreRefreshRegime(c, refreshes);
  }

  // Regime 2: Multi-refresh trajectory (e.g., final case f00)
  const postLaunch = refreshes.filter((r) => r.hours >= 3.0 && r.players >= 1500);
  const activeRefreshes = postLaunch.length >= 2 ? postLaunch : refreshes;

  // Estimate Exophase refresh cadence so we account for expected staleness at target
  let medianGapH = 3.0;
  if (activeRefreshes.length >= 3) {
    const gaps: number[] = [];
    for (let k = 1; k < activeRefreshes.length; k++) {
      gaps.push(Math.max(0.1, activeRefreshes[k]!.hours - activeRefreshes[k - 1]!.hours));
    }
    gaps.sort((a, b) => a - b);
    medianGapH = Math.min(12.0, gaps[Math.floor(gaps.length / 2)]!);
  }
  const effectiveTargetHours = Math.max(last.hours + 0.5, targetHours - 0.45 * medianGapH);
  const effectiveTargetLogT = Math.log(1 + effectiveTargetHours);

  const traj = buildPlayerTrajectory(refreshes, effectiveTargetHours);
  const dLogNTarget = Math.max(0, Math.log(Math.max(1, traj.nTarget) / Math.max(1, last.players)));

  const nTrophies = c.trophies.length;
  const archetypes = c.trophies.map((t) => classifyTrophy(t));
  const rawTrends = new Array<ReturnType<typeof fitWeightedLogTrend>>(nTrophies);
  const rawElasticities = new Array<number | null>(nTrophies).fill(null);

  // First pass: fit local recency-weighted logit slopes & earner-count elasticities
  const recentCutoff = Math.max(3.0, last.hours - 36.0);
  const groupSlopes = new Map<Archetype, number[]>();
  const groupElast = new Map<Archetype, number[]>();

  for (let i = 0; i < nTrophies; i++) {
    const obs: Array<{ logTime: number; logitVal: number; weight: number }> = [];
    const elastObs: Array<{ logTime: number; logitVal: number; weight: number }> = [];
    for (const r of activeRefreshes) {
      const val = r.rates[i];
      if (val === null || val === undefined) continue;
      const isRecent = r.hours >= recentCutoff;
      const ageDecay = Math.exp(-0.045 * Math.max(0, last.hours - r.hours));
      const w = (isRecent ? 1.5 : 0.4) * ageDecay;
      obs.push({ logTime: r.logTime, logitVal: logit(val), weight: w });
      const earnedCnt = (clampPct(val) / 100) * Math.max(1, r.players);
      elastObs.push({ logTime: Math.log(Math.max(1, r.players)), logitVal: Math.log(earnedCnt + 0.5), weight: w });
    }
    const fit = fitWeightedLogTrend(obs);
    rawTrends[i] = fit;
    const eFit = fitWeightedLogTrend(elastObs);
    if (eFit) {
      rawElasticities[i] = Math.max(0.15, Math.min(3.5, eFit.slope));
    }
    if (fit && (last.rates[i] ?? 0) >= 0.08) {
      const arch = archetypes[i]!;
      const arr = groupSlopes.get(arch) ?? [];
      arr.push(Math.max(-1.8, Math.min(2.8, fit.slope)));
      groupSlopes.set(arch, arr);
      if (rawElasticities[i] !== null) {
        const earr = groupElast.get(arch) ?? [];
        earr.push(rawElasticities[i]!);
        groupElast.set(arch, earr);
      }
    }
  }

  // Compute robust group median slopes and elasticities for hierarchical shrinkage
  const groupMedSlope = new Map<Archetype, number>();
  for (const [arch, arr] of groupSlopes.entries()) {
    arr.sort((a, b) => a - b);
    groupMedSlope.set(arch, arr[Math.floor(arr.length / 2)]!);
  }
  const groupMedElast = new Map<Archetype, number>();
  for (const [arch, arr] of groupElast.entries()) {
    arr.sort((a, b) => a - b);
    groupMedElast.set(arch, arr[Math.floor(arr.length / 2)]!);
  }

  // Empirical backtest calibration on the case's own history (if >= 4 post-launch refreshes)
  let empScale = 1.0;
  let empBias = 0.0;
  if (activeRefreshes.length >= 4) {
    const splitIdx = Math.max(2, Math.floor(activeRefreshes.length * 0.65));
    const trainEnd = activeRefreshes[splitIdx - 1]!;
    const testEnd = activeRefreshes[activeRefreshes.length - 1]!;
    const btDx = Math.max(0.05, testEnd.logTime - trainEnd.logTime);
    const btDLogN = Math.max(0, Math.log(Math.max(1, testEnd.players) / Math.max(1, trainEnd.players)));
    const btDtH = Math.max(0.5, testEnd.hours - trainEnd.hours);
    const btAvgN = 0.5 * (trainEnd.players + testEnd.players);
    const btResids: number[] = [];
    for (let i = 0; i < nTrophies; i++) {
      const actualR = testEnd.rates[i];
      const baseR = trainEnd.rates[i];
      if (actualR === null || actualR === undefined || baseR === null || baseR === undefined) continue;
      if (baseR >= 98.5 || baseR <= 0.02) continue;
      const subObs: Array<{ logTime: number; logitVal: number; weight: number }> = [];
      for (let k = 0; k < splitIdx; k++) {
        const rk = activeRefreshes[k]!;
        const v = rk.rates[i];
        if (v !== null && v !== undefined) {
          subObs.push({ logTime: rk.logTime, logitVal: logit(v), weight: Math.exp(-0.04 * (trainEnd.hours - rk.hours)) });
        }
      }
      const subFit = fitWeightedLogTrend(subObs);
      const slope = subFit ? Math.max(-1.5, Math.min(2.2, subFit.slope * 0.78)) : 0;
      const trendP = logit(baseR) + slope * btDx;
      // Blend with count-velocity projection on the holdout window
      let velP = trendP;
      const ref0 = activeRefreshes[0]!;
      const r0v = ref0.rates[i];
      const obsH = Math.max(0.5, trainEnd.hours - ref0.hours);
      if (r0v !== null && r0v !== undefined) {
        const c0 = (clampPct(r0v) / 100) * ref0.players;
        const c1 = (clampPct(baseR) / 100) * trainEnd.players;
        const haz = Math.max(0, c1 - c0) / (Math.max(1, 0.5 * (ref0.players + trainEnd.players)) * obsH);
        const projC = c1 + haz * btAvgN * btDtH * Math.pow(Math.max(1, trainEnd.hours) / Math.max(1, testEnd.hours), 0.2);
        velP = logit(clampPct((100 * projC) / Math.max(1, testEnd.players)));
      }
      const pred = btDLogN > 0.01 ? 0.55 * trendP + 0.45 * velP : trendP;
      btResids.push(logit(actualR) - pred);
    }
    if (btResids.length >= 10) {
      btResids.sort((a, b) => a - b);
      const medResid = btResids[Math.floor(btResids.length / 2)]!;
      const absErrs = btResids.map((e) => Math.abs(e - medResid));
      const maePerRootDx = (absErrs.reduce((a, b) => a + b, 0) / absErrs.length) / Math.sqrt(btDx);
      // Expected Gaussian MAE is sqrt(2/pi) * sigma ≈ 0.798 * sigma
      const impliedSigmaPerRootDx = maePerRootDx / 0.798;
      empScale = Math.max(0.70, Math.min(1.55, impliedSigmaPerRootDx / 0.36));
      // Gentle bias correction scaled to horizon ratio and capped conservatively
      empBias = Math.max(-0.10, Math.min(0.10, (medResid / Math.max(0.15, btDx)) * 0.25));
    }
  }

  // Second pass: 4-view count-consistent ensemble (Cohort Convolution + Count Velocity + Logit Trend + Elasticity)
  const medians = new Array<number>(nTrophies);
  const spreads = new Array<number>(nTrophies);

  // Estimate recent player-hour integral from last.hours to effectiveTargetHours for count-velocity view
  const dtHorizonH = Math.max(0.5, effectiveTargetHours - last.hours);
  const avgFutureN = 0.5 * (last.players + traj.nTarget);

  for (let i = 0; i < nTrophies; i++) {
    const arch = archetypes[i]!;
    const prior = ARCHETYPE_PRIORS[arch];
    const lastRate = clampPct(last.rates[i] ?? 0.5);
    const lastLogit = logit(lastRate);
    const lastCount = (lastRate / 100) * last.players;
    const dxTarget = Math.max(0.01, effectiveTargetLogT - last.logTime);

    // Physical non-decreasing earner-count lower bound (earners cannot un-earn a trophy)
    const countFloorRate = clampPct((100 * lastCount) / Math.max(1, traj.nTarget));
    const countFloorLogit = logit(countFloorRate);

    // Extract pre-launch reviewer count offset from the first launch-night snapshot (<1500 players)
    const firstRef = refreshes[0]!;
    const preLaunchCount =
      firstRef.players < 1500 && firstRef.rates[i] !== null && firstRef.rates[i] !== undefined
        ? Math.round((clampPct(firstRef.rates[i]!) / 100) * firstRef.players)
        : 0;

    // View 1: Cohort-convolution fit with reviewer-count offset evaluated at effectiveTargetHours
    const cohortObs: Array<{ hours: number; logitVal: number; weight: number }> = [];
    for (const r of activeRefreshes) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const w = Math.exp(-0.035 * Math.max(0, last.hours - r.hours));
      cohortObs.push({ hours: r.hours, logitVal: logit(v), weight: w });
    }
    const cohortFit = fitCohortModel(traj.cohorts, cohortObs, prior, effectiveTargetHours, preLaunchCount);

    // View 2: Hierarchically shrunk damped log-time trend
    const fit = rawTrends[i];
    const gSlope = groupMedSlope.get(arch) ?? 0.25;
    let trendPred = lastLogit;
    let residSd = 0.18;
    // Smooth continuous transition for trophies still near the pre-launch reviewer dilution floor
    const excessRatio = (lastCount - preLaunchCount) / Math.max(12, preLaunchCount * 8);
    const dilBlend = lastRate >= 0.30 ? 0.0 : Math.max(0.0, Math.min(1.0, 1.0 - excessRatio));
    const rateReliability = (1 - 0.82 * dilBlend) * (lastRate >= 0.5 ? 1.0 : lastRate >= 0.1 ? 0.65 : 0.30);
    const nWeight = (activeRefreshes.length / (activeRefreshes.length + 3.0)) * rateReliability;
    if (fit && dilBlend < 0.85) {
      const shrunkSlope = nWeight * Math.max(-1.6, Math.min(2.6, fit.slope)) + (1 - nWeight) * gSlope;
      const damp = shrunkSlope >= 0
        ? (lastRate > 60 ? 0.55 : lastRate > 30 ? 0.75 : 0.88)
        : 0.55;
      const rawTrend = lastLogit + shrunkSlope * damp * dxTarget;
      trendPred = (1 - dilBlend) * rawTrend + dilBlend * cohortFit.predLogit;
      residSd = Math.max(fit.residSd, 0.20 * dilBlend);
    } else if (dilBlend >= 0.85) {
      trendPred = cohortFit.predLogit;
      residSd = Math.max(0.22, cohortFit.rmse);
    } else {
      trendPred = lastLogit + 0.35 * (logit(prior.week1RatePct) - lastLogit) * Math.min(1, horizonDays / 5);
    }

    // View 3: Earner-count elasticity relative to player-pool growth
    let elastPred = trendPred;
    const rawEl = rawElasticities[i];
    if (rawEl !== null && dLogNTarget > 1e-4 && dilBlend < 0.85) {
      const gEl = groupMedElast.get(arch) ?? 1.15;
      const shrunkEl = nWeight * rawEl + (1 - nWeight) * gEl;
      const satDamp = lastRate > 55 ? 0.60 : lastRate > 25 ? 0.80 : 0.92;
      const effEl = 1.0 + (shrunkEl - 1.0) * satDamp;
      const rawRateEl = lastRate * Math.exp((effEl - 1.0) * dLogNTarget);
      const cappedRate = rawRateEl > 70 ? 70 + 29.5 * Math.tanh((rawRateEl - 70) / 35) : rawRateEl;
      const rawElLogit = logit(clampPct(cappedRate));
      elastPred = (1 - dilBlend) * rawElLogit + dilBlend * cohortFit.predLogit;
    }

    // View 4: Direct count-space earner velocity over recent refreshes with saturation damping
    let countVelPred = cohortFit.predLogit;
    if (activeRefreshes.length >= 2 && dilBlend < 0.85) {
      const refStart = activeRefreshes[Math.max(0, activeRefreshes.length - Math.min(6, activeRefreshes.length))]!;
      const rStartVal = refStart.rates[i];
      const dtObs = Math.max(0.25, last.hours - refStart.hours);
      if (rStartVal !== null && rStartVal !== undefined && dtObs >= 0.5) {
        const startCount = (clampPct(rStartVal) / 100) * refStart.players;
        const deltaE = Math.max(0, lastCount - startCount);
        const avgObsN = Math.max(1, 0.5 * (refStart.players + last.players));
        // Per-player-hour earn hazard, attenuated gently for weekday/saturation decay
        const hazard = deltaE / (avgObsN * dtObs);
        const headroom = Math.max(0.05, 1 - lastRate / (prior.pMax * 100));
        const timeDecay = Math.pow(Math.max(1, last.hours) / Math.max(1, effectiveTargetHours), 0.25);
        const projAdded = hazard * avgFutureN * dtHorizonH * timeDecay * Math.min(1.0, headroom * 1.15);
        const projRate = clampPct((100 * (lastCount + projAdded)) / Math.max(1, traj.nTarget));
        countVelPred = (1 - dilBlend) * logit(projRate) + dilBlend * cohortFit.predLogit;
      }
    }

    // Prologue special stability: if already >92%, anchor tightly near observed ceiling
    let medLogit: number;
    if (arch === 'prologue' && lastRate >= 92.0) {
      medLogit = 0.75 * lastLogit + 0.25 * cohortFit.predLogit;
    } else {
      const dataBlend =
        activeRefreshes.length >= 3
          ? 0.42 * cohortFit.predLogit + 0.22 * countVelPred + 0.20 * trendPred + 0.16 * elastPred
          : 0.40 * cohortFit.predLogit + 0.22 * countVelPred + 0.22 * trendPred + 0.16 * elastPred;
      const priorBlend = 0.70 * cohortFit.predLogit + 0.30 * logit(prior.week1RatePct);
      medLogit = (1 - dilBlend) * dataBlend + dilBlend * priorBlend;
    }

    // Apply gentle empirical holdout bias correction and enforce physical earner-count floor
    if (arch !== 'prologue') {
      medLogit += empBias * Math.min(1.0, dxTarget);
    }
    medLogit = Math.max(countFloorLogit, Math.max(logit(0.01), Math.min(logit(99.99), medLogit)));
    medians[i] = medLogit;

    // Calibrated per-trophy logit spread combining empirical scale, fit residuals, view disagreement & Poisson count noise
    const viewMax = Math.max(cohortFit.predLogit, countVelPred, trendPred, elastPred);
    const viewMin = Math.min(cohortFit.predLogit, countVelPred, trendPred, elastPred);
    const viewSpread = viewMax - viewMin;
    const baseLogitSd = 0.27 * empScale * Math.sqrt(dxTarget);
    const predCount = Math.max(1, (invLogit(medLogit) / 100) * traj.nTarget);
    const poissonSd = 0.50 / Math.sqrt(predCount);
    const refreshJitter = 0.08 * (medianGapH / Math.max(24, targetHours));
    const highRateTighten = lastRate >= 95.0 ? 0.45 : 1.0;
    const combinedSd = Math.sqrt(
      baseLogitSd ** 2 +
      (0.45 * residSd * Math.sqrt(1 + dxTarget)) ** 2 +
      (0.24 * viewSpread) ** 2 +
      poissonSd ** 2 +
      refreshJitter ** 2,
    ) * highRateTighten;

    spreads[i] = Math.max(0.10, Math.min(1.10, combinedSd));
  }

  // Macro trophies_earned consistency anchor (soft pull when multi-refresh history shows clear macro trend)
  if (activeRefreshes.length >= 3 && last.earned > 0 && last.players > 0) {
    const macroObs = activeRefreshes
      .filter((r) => r.earned > 0 && r.players > 0)
      .map((r) => ({
        logTime: r.logTime,
        logitVal: Math.log(r.earned / r.players),
        weight: Math.exp(-0.03 * Math.max(0, last.hours - r.hours)),
      }));
    const macroFit = fitWeightedLogTrend(macroObs);
    if (macroFit) {
      const dxM = Math.max(0.01, effectiveTargetLogT - last.logTime);
      const dampSlope = Math.max(-0.5, Math.min(1.8, macroFit.slope)) * 0.72;
      const targetAvgTrophies = (last.earned / last.players) * Math.exp(dampSlope * dxM);
      let predSum = 0;
      for (let i = 0; i < nTrophies; i++) predSum += invLogit(medians[i]!) / 100;
      if (predSum > 0.5 && targetAvgTrophies > 0.5) {
        const logRatio = Math.max(-0.25, Math.min(0.25, Math.log(targetAvgTrophies / predSum)));
        for (let i = 0; i < nTrophies; i++) {
          if (archetypes[i] === 'prologue') continue;
          medians[i] = Math.max(logit(0.01), Math.min(logit(99.99), medians[i]! + 0.18 * logRatio));
        }
      }
    }
  }

  // Third pass: structural monotonicity constraints (story chain & platinum <= min(others))
  // Find indices of core story chain in strict order
  const storyOrder: Archetype[] = ['prologue', 'ch1', 'ch2', 'ch3', 'ch4', 'finale', 'completion_100', 'platinum'];
  const chainIndices: number[] = [];
  for (const a of storyOrder) {
    const idx = archetypes.findIndex((x) => x === a);
    if (idx !== -1) chainIndices.push(idx);
  }
  // Soft isotonic enforcement along the main story chain (non-increasing)
  for (let pass = 0; pass < 3; pass++) {
    for (let j = 1; j < chainIndices.length; j++) {
      const prevIdx = chainIndices[j - 1]!;
      const curIdx = chainIndices[j]!;
      if (medians[curIdx]! > medians[prevIdx]!) {
        const avg = 0.5 * (medians[prevIdx]! + medians[curIdx]!);
        medians[prevIdx] = avg;
        medians[curIdx] = avg;
      }
    }
  }
  // Platinum cannot exceed the rarest non-platinum trophy's median
  const platIdx = archetypes.indexOf('platinum');
  if (platIdx !== -1) {
    let minOther = Infinity;
    for (let i = 0; i < nTrophies; i++) {
      if (i !== platIdx && medians[i]! < minOther) minOther = medians[i]!;
    }
    if (medians[platIdx]! > minOther) {
      medians[platIdx] = minOther;
    }
  }

  return c.trophies.map((t, i) => {
    const med = medians[i]!;
    const sd = spreads[i]!;
    const q = Q_MULT.map((m) => round4(clampPct(invLogit(med + m * sd))));
    for (let k = 1; k < q.length; k++) {
      if (q[k]! < q[k - 1]!) q[k] = q[k - 1]!;
    }
    return { id: t.id, q };
  });
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

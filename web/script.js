// web/script.js
// Updates:
// 1) Adds a head circle (user + target) so it feels human.
// 2) More tolerance + easier pass => it will actually auto-cycle.
// 3) Less jumpy: stronger smoothing + tiny “deadzone” + countdown/hold stability.
// 4) Friendlier tips + sweeter, calmer vibe (soft pink, simple guidance).

import * as vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const poseNameEl = document.getElementById("poseName");
const scoreEl = document.getElementById("scorePill");
const streakEl = document.getElementById("streakPill");
const statusEl = document.getElementById("statusPill");
const coachEl = document.getElementById("coachPill");
const hintEl = document.getElementById("hint");

const { FilesetResolver, PoseLandmarker } = vision;

// -------------------- TUNABLES (FRIENDLIER) --------------------
const SMOOTH_ALPHA = 0.35;            // higher = smoother (less jitter)
const DEADZONE_NORM = 0.0025;         // ignore micro movement in normalized coords
const NO_PERSON_RESET_MS = 3500;

const COUNTDOWN_MS = 700;             // faster to feel responsive
const HOLD_MS_DEFAULT = 650;          // easier hold so it progresses
const AUTO_ADVANCE_MS = 850;
const FREEZE_AFTER_SUCCESS_MS = 650;

// Easier match (more tolerance) => will cycle
const PASS_PCT = 0.62;                // was 0.78 (too strict)
const NEAR_PCT = 0.50;                // near cue sooner
const NEAR_SOUND_COOLDOWN = 900;

// If your pose tolerances are too tight, we inflate them here:
const TOLERANCE_BOOST_DEG = 10;       // adds +10° to each tolerance

// target card
const TARGET_CARD = { x: 18, y: 18, w: 200, h: 200 };

// distance coaching
const TORSO_MIN = 0.028;              // a bit looser
const TORSO_MAX = 0.160;

// -------------------- LANDMARKS --------------------
const EDGES = [
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 12],
  [23, 24],
  [11, 23], [12, 24],
  [23, 25], [25, 27],
  [24, 26], [26, 28],
];

// MediaPipe pose has no explicit head joint in Tasks output; we synthesize head from shoulders.
const LM = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28,
};

// -------------------- STATE --------------------
let landmarker = null;
let poses = [];
let poseIdx = 0;

let score = 0;
let streak = 0;

let lastVideoTime = -1;
let lastSeenPersonMs = performance.now();

let smoothLms = null;

let freezeUntil = 0;
let nearToneAt = 0;

let countdownStart = null;
let holdStart = null;

// -------------------- AUDIO --------------------
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
}
function tone(freq, ms, gain = 0.028, type = "sine") {
  try {
    ensureAudio();
    const t0 = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const g = audioCtx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    g.gain.value = gain;
    osc.connect(g);
    g.connect(audioCtx.destination);
    osc.start(t0);
    osc.stop(t0 + ms / 1000);
  } catch {}
}
function startSound() {
  tone(523, 70, 0.02);
  setTimeout(() => tone(659, 80, 0.02), 90);
}
function nearSound() {
  tone(440, 55, 0.018);
}
function successSound() {
  tone(659, 65, 0.03);
  setTimeout(() => tone(880, 90, 0.03), 90);
  setTimeout(() => tone(1175, 120, 0.03), 210);
}

// -------------------- CANVAS SIZING --------------------
function resizeCanvasToStage() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * devicePixelRatio);
  canvas.height = Math.round(rect.height * devicePixelRatio);
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  return { w: rect.width, h: rect.height };
}

function lmToPxMirrored(lm, w, h) {
  return { x: (1 - lm.x) * w, y: lm.y * h };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleABC(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag = (Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y)) + 1e-6;
  const cos = Math.max(-1, Math.min(1, dot / mag));
  return Math.acos(cos) * (180 / Math.PI);
}

// -------------------- ANGLES --------------------
function computeAnglesMirrored(lms, w, h) {
  const p = (i) => lmToPxMirrored(lms[i], w, h);

  const LSh = p(LM.LEFT_SHOULDER), RSh = p(LM.RIGHT_SHOULDER);
  const LE = p(LM.LEFT_ELBOW), RE = p(LM.RIGHT_ELBOW);
  const LW = p(LM.LEFT_WRIST), RW = p(LM.RIGHT_WRIST);

  const LH = p(LM.LEFT_HIP), RH = p(LM.RIGHT_HIP);
  const LK = p(LM.LEFT_KNEE), RK = p(LM.RIGHT_KNEE);
  const LA = p(LM.LEFT_ANKLE), RA = p(LM.RIGHT_ANKLE);

  return {
    left_elbow: angleABC(LSh, LE, LW),
    right_elbow: angleABC(RSh, RE, RW),
    left_knee: angleABC(LH, LK, LA),
    right_knee: angleABC(RH, RK, RA),
    left_hip: angleABC(LSh, LH, LK),
    right_hip: angleABC(RSh, RH, RK),
    left_shoulder: angleABC(LH, LSh, LE),
    right_shoulder: angleABC(RH, RSh, RE),
  };
}

// -------------------- MULTI-PERSON PICK --------------------
function torsoAreaScore(lms) {
  const LS = lms[LM.LEFT_SHOULDER], RS = lms[LM.RIGHT_SHOULDER];
  const LH = lms[LM.LEFT_HIP], RH = lms[LM.RIGHT_HIP];
  if (!LS || !RS || !LH || !RH) return 0;

  const shoulderW = Math.abs(LS.x - RS.x);
  const hipW = Math.abs(LH.x - RH.x);
  const torsoH = Math.abs(((LS.y + RS.y) / 2) - ((LH.y + RH.y) / 2));
  return (shoulderW + hipW) * torsoH;
}

function pickBestPerson(result) {
  const all = result?.landmarks;
  if (!all || !all.length) return null;

  let best = all[0];
  let bestScore = torsoAreaScore(all[0]);
  for (let i = 1; i < all.length; i++) {
    const s = torsoAreaScore(all[i]);
    if (s > bestScore) {
      bestScore = s;
      best = all[i];
    }
  }
  return best;
}

// -------------------- SMOOTHING + DEADZONE --------------------
function smoothLandmarks(next) {
  if (!next) return null;
  if (!smoothLms) {
    smoothLms = next.map(p => ({ x: p.x, y: p.y, z: p.z ?? 0, visibility: p.visibility ?? 1 }));
    return smoothLms;
  }

  for (let i = 0; i < next.length; i++) {
    const a = smoothLms[i];
    const b = next[i];

    const dx = b.x - a.x;
    const dy = b.y - a.y;

    // deadzone: ignore tiny jitter
    if (Math.abs(dx) < DEADZONE_NORM) b.x = a.x;
    if (Math.abs(dy) < DEADZONE_NORM) b.y = a.y;

    a.x = a.x + SMOOTH_ALPHA * (b.x - a.x);
    a.y = a.y + SMOOTH_ALPHA * (b.y - a.y);
    a.z = (a.z ?? 0) + SMOOTH_ALPHA * ((b.z ?? 0) - (a.z ?? 0));
    a.visibility = (a.visibility ?? 1) + SMOOTH_ALPHA * ((b.visibility ?? 1) - (a.visibility ?? 1));
  }
  return smoothLms;
}

// -------------------- FRIENDLY COACHING --------------------
function sizeGuidance(lms) {
  const torso = torsoAreaScore(lms);
  if (torso < TORSO_MIN) return "Step a little closer so I can see you clearly.";
  if (torso > TORSO_MAX) return "Step back a little so your full body fits.";
  return null;
}

function coachingCue(worst) {
  if (!worst) {
    return {
      area: "full body",
      instruction: "Face the camera and match the target silhouette.",
    };
  }

  const area = worst.k.replaceAll("_", " ");
  const needsLargerAngle = worst.actual < worst.target;
  let action;

  if (worst.k.includes("elbow") || worst.k.includes("knee")) {
    action = needsLargerAngle ? "Straighten" : "Bend";
  } else {
    action = needsLargerAngle ? "Open" : "Bring in";
  }

  return {
    area,
    instruction: `${action} your ${area} a little, then hold steady.`,
  };
}

function friendlyHint(pct, worst) {
  if (pct < 0.25) return "Start with your full-body outline, then " + coachingCue(worst).instruction.toLowerCase();
  return coachingCue(worst).instruction;
}

function updateCoach(worst) {
  const cue = coachingCue(worst);
  coachEl.innerHTML = `<strong>Focus</strong>${cue.area}`;
}

// -------------------- SCORING (more tolerant) --------------------
function scorePose(angles, target, tol) {
  const diffs = [];
  for (const k of Object.keys(target)) {
    const a = angles[k];
    if (typeof a !== "number") continue;

    const t = target[k];

    // Boost tolerance globally to make it pass more often
    const allowance = (tol?.[k] ?? 20) + TOLERANCE_BOOST_DEG;

    const d = Math.abs(a - t);
    diffs.push({ k, d, ok: d <= allowance, allowance, actual: a, target: t });
  }
  if (!diffs.length) return { pass: false, pct: 0, worst: null };
  const okCount = diffs.filter(x => x.ok).length;
  const pct = okCount / diffs.length;
  diffs.sort((a, b) => b.d - a.d);
  return { pass: pct >= PASS_PCT, pct, worst: diffs[0] };
}

// -------------------- DRAWING (soft + sweet) --------------------
function roundedRectPath(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

function synthHeadFromShouldersMirrored(lms, w, h) {
  const LS = lmToPxMirrored(lms[LM.LEFT_SHOULDER], w, h);
  const RS = lmToPxMirrored(lms[LM.RIGHT_SHOULDER], w, h);
  const mid = { x: (LS.x + RS.x) / 2, y: (LS.y + RS.y) / 2 };
  const shoulderW = dist(LS, RS);

  // head center slightly above shoulders; radius based on shoulder width
  const radius = clamp(shoulderW * 0.22, 10, 26);
  const center = { x: mid.x, y: mid.y - shoulderW * 0.45 };
  return { center, radius };
}

function drawUserSkeletonWithHead(lms, w, h) {
  // skeleton lines
  ctx.save();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(244, 154, 194, 0.95)"; // soft pink
  ctx.beginPath();
  for (const [a, b] of EDGES) {
    const A = lmToPxMirrored(lms[a], w, h);
    const B = lmToPxMirrored(lms[b], w, h);
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
  }
  ctx.stroke();

  // nodes
  ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
  for (const i of Object.values(LM)) {
    const P = lmToPxMirrored(lms[i], w, h);
    ctx.beginPath();
    ctx.arc(P.x, P.y, 4.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // head circle
  const head = synthHeadFromShouldersMirrored(lms, w, h);
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.strokeStyle = "rgba(244,154,194,0.95)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(head.center.x, head.center.y, head.radius * 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  ctx.restore();
}

function drawTargetPoseCardWithHead(pose) {
  const t = pose?.template_landmarks;
  if (!t) return;

  const card = TARGET_CARD;

  ctx.save();
  // card bg
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.strokeStyle = "rgba(243,198,221,1)";
  ctx.lineWidth = 1.5;
  roundedRectPath(ctx, card.x, card.y, card.w, card.h, 14);
  ctx.fill();
  ctx.stroke();

  // title
  ctx.fillStyle = "rgba(58,33,48,1)";
  ctx.font = "600 13px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  const name = pose?.name ? `Target: ${pose.name}` : "Target Pose";
  ctx.fillText(name, card.x + 12, card.y + 22);

  // inner box
  const inner = { x: card.x + 14, y: card.y + 34, w: card.w - 28, h: card.h - 50 };

  const idxs = Object.values(LM);
  const pts = [];
  for (const idx of idxs) {
    const pt = t[String(idx)];
    if (!pt) continue;
    // mirror to match camera
    pts.push({ idx, x: 1 - pt[0], y: pt[1] });
  }
  if (!pts.length) { ctx.restore(); return; }

  // bounds
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }
  const pad = 0.08;
  minX = clamp(minX - pad, 0, 1);
  minY = clamp(minY - pad, 0, 1);
  maxX = clamp(maxX + pad, 0, 1);
  maxY = clamp(maxY + pad, 0, 1);

  const s = Math.min(
    inner.w / Math.max(1e-6, (maxX - minX)),
    inner.h / Math.max(1e-6, (maxY - minY))
  );
  const ox = inner.x + (inner.w - (maxX - minX) * s) / 2;
  const oy = inner.y + (inner.h - (maxY - minY) * s) / 2;

  const mapT = (idx) => {
    const pt = t[String(idx)];
    if (!pt) return null;
    const x = 1 - pt[0];
    const y = pt[1];
    return { x: ox + (x - minX) * s, y: oy + (y - minY) * s };
  };

  // draw edges
  ctx.strokeStyle = "rgba(244,154,194,0.95)";
  ctx.lineWidth = 4;
  ctx.beginPath();
  for (const [a, b] of EDGES) {
    const A = mapT(a);
    const B = mapT(b);
    if (!A || !B) continue;
    ctx.moveTo(A.x, A.y);
    ctx.lineTo(B.x, B.y);
  }
  ctx.stroke();

  // nodes
  ctx.fillStyle = "rgba(244,154,194,0.9)";
  for (const idx of idxs) {
    const P = mapT(idx);
    if (!P) continue;
    ctx.beginPath();
    ctx.arc(P.x, P.y, 4.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // head (synthesized from shoulder midpoint)
  const LS = mapT(LM.LEFT_SHOULDER);
  const RS = mapT(LM.RIGHT_SHOULDER);
  if (LS && RS) {
    const mid = { x: (LS.x + RS.x) / 2, y: (LS.y + RS.y) / 2 };
    const shoulderW = dist(LS, RS);
    const radius = clamp(shoulderW * 0.22, 8, 22);
    const headC = { x: mid.x, y: mid.y - shoulderW * 0.45 };

    ctx.fillStyle = "rgba(255,255,255,0.85)";
    ctx.strokeStyle = "rgba(244,154,194,0.95)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(headC.x, headC.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // bottom helper text
  ctx.fillStyle = "rgba(107,63,86,1)";
  ctx.font = "500 12px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  ctx.fillText("Match the card. Hold still to win.", card.x + 12, card.y + card.h - 12);

  ctx.restore();
}

function drawHoldUI(mode, progress01, w, h) {
  const barW = Math.min(520, w - 40);
  const barH = 12;
  const x = (w - barW) / 2;
  const y = h - 26;

  ctx.save();

  // bg
  ctx.fillStyle = "rgba(255,255,255,0.70)";
  ctx.strokeStyle = "rgba(243,198,221,1)";
  ctx.lineWidth = 1.2;
  roundedRectPath(ctx, x, y, barW, barH, 999);
  ctx.fill();
  ctx.stroke();

  // fill
  const fill = clamp(progress01, 0, 1) * barW;
  ctx.fillStyle = "rgba(244,154,194,0.95)";
  roundedRectPath(ctx, x, y, fill, barH, 999);
  ctx.fill();

  // label
  ctx.fillStyle = "rgba(58,33,48,1)";
  ctx.font = "700 13px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  const label = mode === "countdown" ? "Almost… stay still" : mode === "hold" ? "Hold it…" : "";
  if (label) ctx.fillText(label, x, y - 8);

  ctx.restore();
}

function drawCenterToast(text, w, h) {
  ctx.save();
  const bw = Math.min(560, w - 40);
  const bh = 64;
  const bx = (w - bw) / 2;
  const by = (h - bh) / 2;

  ctx.fillStyle = "rgba(255,255,255,0.78)";
  ctx.strokeStyle = "rgba(243,198,221,1)";
  ctx.lineWidth = 1.4;
  roundedRectPath(ctx, bx, by, bw, bh, 16);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(58,33,48,1)";
  ctx.font = "800 18px system-ui, -apple-system, Segoe UI, Roboto, Arial";
  const tw = ctx.measureText(text).width;
  ctx.fillText(text, bx + (bw - tw) / 2, by + 40);

  ctx.restore();
}

// -------------------- SETUP --------------------
async function loadPoses() {
  const res = await fetch("/api/poses");
  const data = await res.json();
  poses = data.poses || [];
}

async function setupCamera() {
  statusEl.textContent = "Status: camera";
  hintEl.textContent = "Allow camera access. Then step into the frame.";
  updateCoach(null);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

async function setupLandmarker() {
  statusEl.textContent = "Status: loading";
  hintEl.textContent = "Loading pose tracking…";

  const wasmRoot = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
  const fileset = await FilesetResolver.forVisionTasks(wasmRoot);

  const modelUrl =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: modelUrl },
    runningMode: "VIDEO",
    numPoses: 3,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  statusEl.textContent = "Status: ready";
  hintEl.textContent = "Step into frame. Try to show your upper body clearly.";
}

function setActivePose(i) {
  if (!poses.length) return;
  poseIdx = (i + poses.length) % poses.length;
  const p = poses[poseIdx];

  poseNameEl.textContent = `Pose ${poseIdx + 1}/${poses.length}: ${p?.name ?? "—"}`;

  countdownStart = null;
  holdStart = null;
  freezeUntil = performance.now() + 250;

  hintEl.textContent = "Match the target card (top-left). Move slowly. Hold still to win.";
  updateCoach(null);
}

function advancePose() {
  setActivePose(poseIdx + 1);
  statusEl.textContent = "Status: next";
}

function markSuccess() {
  score += (poses[poseIdx]?.points ?? 10);
  streak += 1;
  scoreEl.textContent = `Score ${score}`;
  streakEl.textContent = `Streak ${streak}`;

  statusEl.textContent = "Status: success";
  hintEl.textContent = "Sweet! Next pose…";
  coachEl.innerHTML = "<strong>Focus</strong>matched";

  successSound();

  freezeUntil = performance.now() + FREEZE_AFTER_SUCCESS_MS;
  setTimeout(() => advancePose(), AUTO_ADVANCE_MS);
}

// -------------------- LOOP --------------------
function loop() {
  if (!landmarker) return requestAnimationFrame(loop);
  if (video.currentTime === lastVideoTime) return requestAnimationFrame(loop);
  lastVideoTime = video.currentTime;

  const now = performance.now();
  const { w, h } = resizeCanvasToStage();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const pose = poses[poseIdx];
  if (!pose) {
    statusEl.textContent = "Status: no poses";
    hintEl.textContent = "No poses found. Check /api/poses.";
    return requestAnimationFrame(loop);
  }

  // draw target card every frame
  drawTargetPoseCardWithHead(pose);

  if (now < freezeUntil) {
    drawHoldUI("none", 0, w, h);
    return requestAnimationFrame(loop);
  }

  const result = landmarker.detectForVideo(video, now);
  const bestRaw = pickBestPerson(result);

  if (!bestRaw) {
    statusEl.textContent = "Status: find you";
    hintEl.textContent = "Step into view. Face the camera. Good lighting helps.";
    updateCoach(null);
    drawCenterToast("Step into the frame", w, h);

    countdownStart = null;
    holdStart = null;

    if (now - lastSeenPersonMs > NO_PERSON_RESET_MS) {
      smoothLms = null;
      freezeUntil = now + 350;
    }
    return requestAnimationFrame(loop);
  }

  lastSeenPersonMs = now;

  const lms = smoothLandmarks(bestRaw);

  // draw user skeleton + head
  drawUserSkeletonWithHead(lms, w, h);

  // distance guidance
  const guide = sizeGuidance(bestRaw);
  if (guide) {
    statusEl.textContent = "Status: adjust";
    hintEl.textContent = guide;
    coachEl.innerHTML = "<strong>Focus</strong>framing";
  }

  // score
  const angles = computeAnglesMirrored(lms, w, h);
  const scored = scorePose(angles, pose.target_angles || {}, pose.tolerances || {});
  const pct = scored.pct;

  // gentle near cue
  if (!scored.pass && pct >= NEAR_PCT && (now - nearToneAt) > NEAR_SOUND_COOLDOWN) {
    nearSound();
    nearToneAt = now;
  }

  // game flow
  if (scored.pass) {
    // countdown
    if (countdownStart == null && holdStart == null) {
      countdownStart = now;
      statusEl.textContent = "Status: almost";
      hintEl.textContent = "Perfect! Stay still…";
      coachEl.innerHTML = "<strong>Focus</strong>hold steady";
    }

    if (countdownStart != null) {
      const cdProg = clamp((now - countdownStart) / COUNTDOWN_MS, 0, 1);
      drawHoldUI("countdown", cdProg, w, h);

      if (cdProg >= 1) {
        countdownStart = null;
        holdStart = now;
        statusEl.textContent = "Status: hold";
        hintEl.textContent = "Hold it… just a moment!";
      }
    }

    if (holdStart != null) {
      const need = pose.hold_ms ?? HOLD_MS_DEFAULT;
      const holdProg = clamp((now - holdStart) / need, 0, 1);
      drawHoldUI("hold", holdProg, w, h);

      statusEl.textContent = `Status: hold ${Math.round(holdProg * 100)}%`;

      if (holdProg >= 1) {
        holdStart = null;
        markSuccess();
      }
    }
  } else {
    countdownStart = null;
    holdStart = null;

    const pctText = Math.round(pct * 100);
    statusEl.textContent = `Status: ${pctText}%`;
    updateCoach(scored.worst);
    hintEl.textContent = friendlyHint(pct, scored.worst);
    drawHoldUI("none", 0, w, h);
  }

  requestAnimationFrame(loop);
}

// -------------------- START --------------------
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;

  try {
    ensureAudio();
    if (audioCtx.state === "suspended") await audioCtx.resume();

    startSound();

    scoreEl.textContent = `Score ${score}`;
    streakEl.textContent = `Streak ${streak}`;

    await loadPoses();
    await setupCamera();
    await setupLandmarker();

    setActivePose(0);
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Status: error";
    hintEl.textContent = "Something failed. Open the Console for details.";
    startBtn.disabled = false;
  }
});

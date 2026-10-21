// Optional decoration (the background animation of the original UI); the chat works without it.
/* mindmeld.js: the "AI mind-meld" ASCII animation (start-up panel, and full screen behind the sign-in card).
   A halftone brain (lateral view facing left: frontal, parietal, temporal and occipital lobes, cerebellum,
   brainstem; anatomical sulci traced by hand and left as gaps; every gyrus shaded like a tube with a character
   density ramp) with molecules, proteins, DNA, galaxies, atoms and stars orbiting it on inclined orbits (hidden
   while behind it). Cycle: orbit -> accelerating spiral in with trails -> collapse -> brain flash -> burst of
   sparks -> objects re-form. Between collapses, activity travels: signals run along pathways from one cortical
   region to a neighboring one (bright head, fading trail) and may fire on into the next on arrival. The brain is an original procedural design in the style of halftone "ASCII brain"
   illustrations; no artist's piece is copied, so there is no signature to keep.
   Plain JS, no dependencies, no network. ~20 fps; does no work while the tab or the element is hidden.
   API: Mindmeld.mount(preElement, opts?) -> { setProgress(0..1), finale(), stop() }
     opts.cols / opts.rows: grid size (default 120 x 28); opts.fit: true sizes the grid to the element (full screen);
     opts.progress: initial intensity. Honors prefers-reduced-motion (one still frame). */
(function () {
  'use strict';
  const RAMP = ' .·:-=+*#%@';
  const BRAMP = ' .:+*#@@@@@';                // the brain's own ramp: few levels, so gyri read as solid
  const T_ORBIT = 5.4, T_IN = 1.9, T_BURST = 1.7, T_CYCLE = T_ORBIT + T_IN + T_BURST;
  const rnd = (a, b) => a + Math.random() * (b - a);
  const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
  const esc = (c) => (c === '<' ? '&lt;' : c === '>' ? '&gt;' : c === '&' ? '&amp;' : c);

  // ---- the brain outline and folds, traced in a 100 x 76 unit box (x right, y down; frontal pole at x = 0) -----
  const CEREBRUM = [[2, 38], [4, 27], [11, 15], [22, 7], [36, 2], [52, 1], [67, 3], [80, 8], [90, 16], [96, 26],
    [99, 36], [98, 44], [93, 49], [84, 51], [76, 53], [66, 57], [54, 60], [42, 61], [33, 59], [27, 54], [24, 49],
    [17, 47], [9, 46], [4, 43]];
  const CEREBELLUM = [[67, 62], [74, 57], [85, 56], [95, 58], [99, 64], [96, 70], [87, 74], [76, 73], [69, 68]];
  const STEM = [[55, 59], [64, 58], [66, 65], [68, 76], [61, 76], [58, 66]];
  const SULCI = [
    [[24, 47], [33, 41], [47, 37], [60, 34], [67, 27]],             // lateral (Sylvian) fissure
    [[53, 2], [50, 13], [47, 24], [44, 34]],                         // central sulcus
    [[42, 4], [39, 15], [36, 26], [34, 34]],                         // precentral
    [[63, 5], [60, 16], [57, 26], [55, 33]],                         // postcentral
    [[38, 11], [27, 13], [16, 19]],                                  // superior frontal
    [[35, 23], [24, 25], [11, 31]],                                  // inferior frontal
    [[61, 19], [74, 21], [86, 29]],                                  // intraparietal
    [[73, 7], [77, 14]],                                             // parietal fold
    [[29, 52], [44, 47], [62, 43], [80, 39], [91, 36]],              // superior temporal
    [[37, 57], [55, 53], [74, 49]],                                  // inferior temporal
    [[86, 13], [91, 24]],                                            // parieto-occipital
    [[94, 31], [92, 43]],                                            // lunate
    [[17, 39], [26, 34]],                                            // orbital (the rest: only on larger brains)
    [[22, 8], [19, 15]], [[13, 25], [7, 29]], [[6, 34], [12, 37]],   // frontal minor folds
    [[30, 17], [27, 21]], [[66, 10], [65, 16]],                      // pre/postcentral branches
    [[64, 31], [71, 26], [78, 30]], [[82, 33], [88, 30]],            // supramarginal / angular
    [[84, 43], [93, 47]], [[48, 51], [51, 57]], [[66, 47], [69, 52]], // occipital, temporal minor folds
  ];
  // smooth 2-D gradient noise for the fine folds at large sizes (seeded, deterministic)
  const PERM = new Uint8Array(512);
  (function () {
    let s = 20261001; const p = Array.from({ length: 256 }, (_, i) => i);
    for (let i = 255; i > 0; i--) { s = (s * 1103515245 + 12345) & 0x7fffffff; const j = s % (i + 1); [p[i], p[j]] = [p[j], p[i]]; }
    for (let i = 0; i < 512; i++) PERM[i] = p[i & 255];
  })();
  const grad = (h, x, y) => { const a = (h / 256) * Math.PI * 2; return Math.cos(a) * x + Math.sin(a) * y; };
  function noise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = PERM[(PERM[xi & 255] + yi) & 511], b = PERM[(PERM[(xi + 1) & 255] + yi) & 511];
    const c = PERM[(PERM[xi & 255] + yi + 1) & 511], d = PERM[(PERM[(xi + 1) & 255] + yi + 1) & 511];
    const x1 = grad(a, xf, yf) + u * (grad(b, xf - 1, yf) - grad(a, xf, yf));
    const x2 = grad(c, xf, yf - 1) + u * (grad(d, xf - 1, yf - 1) - grad(c, xf, yf - 1));
    return x1 + v * (x2 - x1);
  }
  // traveling activity: cortical regions (nodes, unit coords, each on a gyrus) and the pathways between them.
  // A signal runs from one region through up to 3-5 pathways, never turning more than 40 degrees (brightest head, fading trail); on arrival
  // it may fire on.
  const NODES = [[12, 32], [20, 18], [32, 9], [43, 20], [53, 20], [68, 12], [72, 30], [92, 38], [72, 45], [40, 44],
    [24, 30], [46, 6], [84, 22], [58, 39], [76, 64], [93, 64]];
  // 0 prefrontal, 1 frontal, 2 superior frontal, 3 motor, 4 sensory, 5 superior parietal, 6 inferior parietal,
  // 7 occipital, 8 posterior temporal, 9 anterior temporal, 10 inferior frontal, 11 vertex, 12 parieto-occipital,
  // 13 insular / mid temporal, 14-15 cerebellum
  const TRACTS = [[0, 1], [1, 2], [2, 3], [2, 11], [11, 4], [3, 4], [3, 10], [0, 10], [4, 5], [5, 12], [5, 6], [4, 6],
    [6, 12], [12, 7], [6, 8], [7, 8], [8, 13], [13, 9], [9, 10], [10, 13], [8, 9], [3, 13], [1, 10], [14, 15]];
  const ADJ = NODES.map((_, n) => TRACTS.map((e, q) => (e[0] === n || e[1] === n ? q : -1)).filter((q) => q >= 0));
  const N_MAJOR = 9;                          // the first 9 sulci are drawn at every size
  function inPoly(P, x, y) {
    let c = false;
    for (let a = 0, b = P.length - 1; a < P.length; b = a++) {
      const [xa, ya] = P[a], [xb, yb] = P[b];
      if ((ya > y) !== (yb > y) && x < ((xb - xa) * (y - ya)) / (yb - ya) + xa) c = !c;
    }
    return c;
  }

  // ---- orbiter sprites. Large (near, full screen >= 110 columns) ones are drawn with String.raw so backslashes stay literal
  const sp = (str) => str.split('\n').filter((r, q, a) => !(q === 0 && !r.trim()) && !(q === a.length - 1 && !r.trim()));
  // Own designs (no artist's piece copied), sized ~65% of the first full-screen set after review.
  const BIG = [
    { k: 'mol', f: [sp(String.raw`
   ___
  / _ \
 / / \ \
 \ \_/ /
  \___/`)] },                                                        // 0 benzene: hexagon + aromatic ring
    { k: 'mol', f: [sp(String.raw`
    (O)
   /   \
 (H)   (H)`)] },                                                     // 1 water, ball and stick
    { k: 'mol', f: [sp(String.raw`
 .-.   .-.   .-.
( O )=( C )=( O )
 '-'   '-'   '-'`)] },                                               // 2 carbon dioxide
    { k: 'prot', f: [sp(String.raw`
 .--. .--. .--.
/ .-'/ .-'/ .-'\
\ '-.\ '-.\ '-./
 '--' '--' '--'`), sp(String.raw`
 .--. .--. .--.
/'-. /'-. /'-. \
\.-' \.-' \.-' /
 '--' '--' '--'`)] },                                                // 3 alpha-helix ribbon
    { k: 'prot', f: [sp(String.raw`
  __    __    __
 /  \  /  \  /  \
/    \/    \/    \=>`), sp(String.raw`
 __    __    __
/  \  /  \  /  \
    \/    \/    \/=>`)] },                                           // 4 ribbon wave ending in a strand arrow
    { k: 'dna', f: [sp(String.raw`
\  .-=-.  .-=-.  /
 \/ | | \/ | | \/
 /\ | | /\ | | /\
/  '-=-'  '-=-'  \ `), sp(String.raw`
 .-=-.  .-=-.  .-
/ | | \/ | | \/ |
\ | | /\ | | /\ |
 '-=-'  '-=-'  '-`)] },                                              // 5 DNA double helix with base pairs
    { k: 'gal', f: [sp(String.raw`
 '.     .-~~-.
   '-.-'  _   '.
   .'   .(@).   '.
    '.   '~'  .-'
      '-.__.-'-.   '.`), sp(String.raw`
 '.     .-~~-.
   '-.-'  .   '.
   .'   .(@).   '.
    '.   '*'  .-'
      '-.__.-'-.   '.`)] },                                          // 6 spiral galaxy, two trailing arms
    { k: 'atom', f: [sp(String.raw`
   _..._
 .' / \ '.
(--( @ )--)
 '. \_/ .'
   '...'`), sp(String.raw`
   _..._
 .' / \ '.
( -( @ )- )
 '. \_/ .'
   '...'`)] },                                                       // 7 atom: nucleus + three orbits
    { k: 'star', f: [sp(String.raw`
  .
-=*=-
  '`), sp(String.raw`
\ | /
- * -
/ | \ `), sp(String.raw`
  |
--+--
  |`)] },                                                            // 8 star
  ];

  // ---- orbiter families: four levels of detail each (tiny, small, medium, large); each level is a list of frames,
  // each frame a list of rows. Depth picks the level: far -> small, near -> large (see LODS in mount()).
  const F = (k, tiny, small, medium, large) => ({ k, lv: [tiny, small, medium, large] });
  const FAMILIES = [
    F('mol', [['o']], [['<O>']], [[' __ ', '/()\\', '\\__/']], BIG[0].f),                               // benzene
    F('mol', [['·']], [['H-O-H']], [['  O  ', ' / \\ ', 'H   H']], BIG[1].f),                            // water
    F('mol', [['-']], [['O=C=O']], [['(O)=(C)=(O)']], BIG[2].f),                                        // CO2
    F('mol', [['·']], [['CH4']], [['  H  ', 'H-C-H', '  H  ']],
      [sp(String.raw`
      H
      |
  H---C---H
      |
      H`)]),                                                                                             // methane
    F('prot', [['~']], [['/\\/\\'], ['\\/\\/']], [['~(/)(/)(/)~'], ['~(\\)(\\)(\\)~']], BIG[3].f),        // helix
    F('prot', [['>']], [['~~>']], [['/\\/\\/\\=>'], ['\\/\\/\\/=>']], BIG[4].f),                          // coil/strand
    F('dna', [['x']], [['\\/\\/'], ['/\\/\\']], [['\\_/~\\_/~', '/~\\_/~\\_'], ['_/~\\_/~\\', '~\\_/~\\_/']], BIG[5].f),
    F('gal', [['@']], [['-=(@)=-'], ['=-(@)-=']], [['  .-~\'', ' ( @ )', '\'~-\'  '], ['\'~-.  ', ' ( @ )', '  `-~\'']], BIG[6].f),
    F('atom', [['o']], [['(o)']], [['·(o) '], [' (o)·'], [' (o) ']], BIG[7].f),
    F('star', [['·']], [['*'], ['+'], ['·'], ['+']], [['+'], ['*'], ['+']], BIG[8].f),
    F('star', [['.']], [['+'], ['·'], ['*'], ['·']], [['*'], ['+'], ['*']], [['\\|/', '-*-', '/|\\'], ['.|.', '-+-', '\'|\'']]),
  ];

  // ---- a scene: grid size, brain raster, orbit geometry ----------------------------------------------------------
  function buildScene(W, H, aspect, portrait, fill) {
    // portrait full screen (a phone): the brain sits in the top part, above a centered card
    const CX = W / 2, CY = portrait ? Math.max(H * 0.2, (W * 0.8 * 76) / 100 / aspect * 0.62) : H / 2;
    // brain width in columns: ~48% of the width, and short enough to leave orbit room above and below
    const BCOLS = Math.max(24, Math.round(portrait ? W * 0.8 : Math.min(W * 0.48, (H * fill * 100 * aspect) / 76)));
    const k = BCOLS / 46;                     // detail scale relative to the panel version
    const UX = 100 / BCOLS, UY = UX * aspect; // units per column / per row
    const BX0 = CX - BCOLS / 2, BY0 = CY - 0.5 - 37 / UY, BCY = BY0 + 33 / UY;
    const segDist = (px, py, ax, ay, bx, by) => {                     // in cell widths (rows scaled by aspect)
      const sx = 1 / UX, sy = aspect / UY;
      px *= sx; ax *= sx; bx *= sx; py *= sy; ay *= sy; by *= sy;
      const dx = bx - ax, dy = by - ay, L = dx * dx + dy * dy, u = clamp(((px - ax) * dx + (py - ay) * dy) / (L || 1), 0, 1);
      return Math.hypot(px - ax - u * dx, py - ay - u * dy);
    };
    const edgeDist = (P, x, y) => { let m = 1e9; for (let a = 0, b = P.length - 1; a < P.length; b = a++) m = Math.min(m, segDist(x, y, ...P[a], ...P[b])); return m; };
    const gap = 0.45 * Math.pow(k, 0.7), tubeW = 1.0 * k;
    const BRAIN = [], MASK = Array.from({ length: H }, () => new Uint8Array(W));
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const x = (i + 0.5 - BX0) * UX, y = (j + 0.5 - BY0) * UY;
      if (x < 0 || x > 100 || y < 0 || y > 77) continue;
      let part = null, d = 0;
      if (inPoly(CEREBRUM, x, y)) {
        part = 'c';
        let sd = 1e9;
        const wx = x + 4.5 * noise(x * 0.12, y * 0.12 + 4), wy = y + 4.5 * noise(x * 0.12 + 11, y * 0.12); // wavy folds
        for (let li = 0; li < (k < 1.2 ? N_MAJOR : SULCI.length); li++) { const L = SULCI[li]; for (let q = 1; q < L.length; q++) sd = Math.min(sd, segDist(wx, wy, ...L[q - 1], ...L[q])); }
        // each gyrus shaded like a tube: densest along its crown, fading to a gap at the sulci and at the rim
        const light = clamp(1.02 - 0.0022 * x - 0.0035 * y, 0, 1);
        const tube = Math.min(clamp((sd - gap * 0.8) / tubeW, 0, 1), clamp(edgeDist(CEREBRUM, x, y) / (1.6 * k), 0, 1) * 0.7 + 0.3);
        d = sd < gap ? 0 : light * (0.5 + 0.5 * Math.sqrt(tube));
        if (k >= 1.4 && d > 0) {                                        // fine folds: thin, lighter channels
          const fn = (X, Y) => noise(X * 0.16 + 0.5 * noise(X * 0.05, Y * 0.05), Y * 0.11 + 0.5 * noise(X * 0.05 + 9, Y * 0.05));
          const n0 = fn(x, y), e = UX * 0.6;
          if (n0 * fn(x + e, y) < 0 || n0 * fn(x, y + e * aspect) < 0) d = 0.12;
        }
      } else if (inPoly(CEREBELLUM, x, y)) {
        part = k < 1.4 ? 'cb' : 'c';                                   // folia: thin horizontal gaps
        const fol = Math.abs(Math.sin((y - 0.12 * (x - 80)) * 0.62 * Math.max(1, k * 0.8)));
        d = fol < 0.3 ? 0 : 0.85 * Math.sqrt(clamp(edgeDist(CEREBELLUM, x, y) / (1.4 * k), 0, 1) * 0.7 + 0.3) * (0.5 + 0.5 * fol);
      } else if (inPoly(STEM, x, y)) {
        part = 'st'; d = 0.25 + 0.5 * clamp(edgeDist(STEM, x, y) / (1.2 * k), 0, 1);
      }
      if (part) { BRAIN.push({ i, j, x, y, d: clamp(d, 0, 1), part }); MASK[j][i] = 1; }
    }
    const out = (i, j) => !(MASK[j] && MASK[j][i]);
    const EDGE = BRAIN.filter((c) => c.part === 'c' && (out(c.i, c.j - 1) || out(c.i, c.j + 1) || out(c.i - 1, c.j) || out(c.i + 1, c.j)));
    const ORX = portrait ? W / 2 - 2 : Math.min(W / 2 - 3, BCOLS * 0.98);                     // orbit radius in columns
    const incMax = Math.min(0.5, Math.asin(clamp(((Math.min(CY, H - CY) - 1.5) * aspect) / ORX, 0.05, 1)));
    const STARS = Array.from({ length: Math.round((W * H) / 70) }, (_, q) => ({ x: (q * 37 + 11) % W, y: (q * 23 + 5 + Math.floor(q / W) * 7) % H, ph: q * 2.3 }));
    return { W, H, CX, CY, aspect, k, BRAIN, MASK, EDGE, BCY, ORX, incMax, STARS, BX0, BY0, UX, UY };
  }

  function mount(pre, opts) {
    const noop = { setProgress() {}, finale() {}, stop() {} };
    if (!pre) return noop;
    opts = opts || {};
    const reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    let S = null, objs = [], sparks = [], pulses = [], signals = [], raf = 0, last = 0, progress = clamp(+opts.progress || 0, 0, 1), stopped = false;
    // full screen: orbits ~30% slower, with the whole orbit / collapse / burst cycle stretched to match
    const TS = opts.fit ? 1.45 : 1, tO = T_ORBIT * TS, tI = T_IN * TS, tC = T_CYCLE * TS;
    // levels of detail used for far / middle / near: full screen on a wide grid uses small / medium / large,
    // the panel and phones tiny / small / medium (set per scene in ensureScene)
    let LODS = [0, 1, 2];
    let t = 0, cycStart = 0, phase = 'orbit', flashUntil = 0, finaleReq = false, big = false, sizeKey = '';

    function measure() {                      // character cell size of this <pre> (falls back to a typical mono)
      const probe = document.createElement('span');        // 10 columns x 4 lines: width and line height
      probe.style.cssText = 'display:inline-block;position:absolute;visibility:hidden';
      probe.textContent = 'MMMMMMMMMM\nM\nM\nM';
      pre.appendChild(probe);
      const r = probe.getBoundingClientRect();
      probe.remove();
      const cw = r.width / 10 || 6, chh = r.height / 4 || 10;
      return { cw, chh, aspect: clamp(chh / cw, 1.3, 2.4) };
    }
    function ensureScene() {
      const m = measure();
      let W = opts.cols || 120, H = opts.rows || 28;
      if (opts.fit) { W = Math.max(40, Math.floor(pre.clientWidth / m.cw)); H = Math.max(16, Math.floor(pre.clientHeight / m.chh)); }
      const key = `${W}x${H}@${m.aspect.toFixed(2)}`;
      if (key === sizeKey) return;
      LODS = opts.fit && W >= 110 ? [1, 2, 3] : [0, 1, 2];
      sizeKey = key; S = buildScene(W, H, m.aspect, !!opts.fit && H * m.aspect > W * 1.15, opts.fit ? 0.8 : 0.9); pulses = []; sparks = []; signals = [];
      spawn(1); objs.forEach((o) => { o.fade = 1; });
    }

    function makeObj(i, n, rOut) {
      const kd = FAMILIES[(i * 7 + Math.floor(Math.random() * 3)) % FAMILIES.length];
      const th = (i / n) * Math.PI * 2 + rnd(-0.25, 0.25), zn = Math.sin(th);
      return { kd, th, R: rnd(0.8, 1.0), inc: rnd(0.2, 1) * S.incMax * (Math.random() < 0.5 ? -1 : 1),
               roll: rnd(-0.18, 0.18), w: rnd(0.45, 0.75), fr: Math.floor(rnd(0, 4)), r: rOut, trail: [], alive: true, fade: 0, lod: zn > 0.3 ? 2 : zn < -0.35 ? 0 : 1 };
    }
    function spawn(rOut) {
      const n = Math.round((8 + 8 * progress + (big ? 3 : 0)) * Math.max(1, Math.sqrt(S.k)) * (opts.fit ? 0.8 : 1));
      objs = Array.from({ length: n }, (_, i) => makeObj(i, n, rOut));
    }
    function pos(o) {                                                    // inclined circular orbit, projected
      const x3 = Math.cos(o.th) * o.R * o.r, z3 = Math.sin(o.th) * o.R * o.r;
      const y = -z3 * Math.sin(o.inc), z = z3 * Math.cos(o.inc);
      const xr = x3 * Math.cos(o.roll) - y * Math.sin(o.roll), yr = x3 * Math.sin(o.roll) + y * Math.cos(o.roll);
      return { x: S.CX + xr * S.ORX, y: S.CY + (yr * S.ORX) / S.aspect, z };
    }
    function burst(n, power) {
      const E = S.EDGE;
      for (let s = 0; s < n; s++) {
        const c = E[Math.floor(Math.random() * E.length)];
        const a = Math.atan2((c.j - S.BCY) * S.aspect, c.i - S.CX) + rnd(-0.35, 0.35), v = rnd(10, 30) * power * Math.sqrt(S.k);
        const life = rnd(0.6, 1.3) * (0.8 + 0.3 * power);
        sparks.push({ x: c.i, y: c.j, vx: Math.cos(a) * v, vy: (Math.sin(a) * v) / S.aspect, life, max: life,
                      cls: ['sp', 'sp', 'sp2', 'mol', 'prot'][Math.floor(Math.random() * 5)] });
      }
    }

    // ---- traveling activity -----------------------------------------------------------------------------------
    const COS_TURN = Math.cos((30 * Math.PI) / 180);   // 30 per point keeps every visible bend under 40 degrees, measured
    const other = (e, n) => (TRACTS[e][0] === n ? TRACTS[e][1] : TRACTS[e][0]);
    function turnOK(p0, p1, p2) {             // turn at p1 from segment p0->p1 to p1->p2 is at most 40 degrees
      const ax = p1[0] - p0[0], ay = p1[1] - p0[1], bx = p2[0] - p1[0], by = p2[1] - p1[1];
      return ax * bx + ay * by >= COS_TURN * Math.hypot(ax, ay) * Math.hypot(bx, by);
    }
    // interior points of one pathway: an overall bow plus a small alternating zigzag (3 points), every turn <= 40
    // degrees, including the turn at `p0` when it continues from a previous point `pp`; amplitude shrinks on retry
    function wiggle(n0, n1, pp) {
      const p0 = NODES[n0], p1 = NODES[n1], dx = p1[0] - p0[0], dy = p1[1] - p0[1], L = Math.hypot(dx, dy);
      const nx = -dy / L, ny = dx / L;
      for (let tr = 0, amp = 1; tr < 8; tr++, amp *= 0.7) {
        const bow = rnd(-0.22, 0.22) * L * amp, zig = rnd(0.05, 0.09) * L * amp * (Math.random() < 0.5 ? 1 : -1);
        const pts = [0.25, 0.5, 0.75].map((u, k) => {
          const o = bow * 4 * u * (1 - u) + zig * (k % 2 ? -1 : 1);
          return [p0[0] + dx * u + nx * o, p0[1] + dy * u + ny * o];
        });
        const Q = [pp || null, p0, ...pts, p1].filter(Boolean);
        let ok = true;
        for (let k = 1; k < Q.length - 1 && ok; k++) ok = turnOK(Q[k - 1], Q[k], Q[k + 1]);
        if (ok) return pts;
      }
      return null;
    }
    function splineCells(P) {                 // Catmull-Rom through the polyline P (unit coords): smooth, no corners
      const cells = [];
      for (let k = 0; k < P.length - 1; k++) {
        const p0 = P[Math.max(0, k - 1)], p1 = P[k], p2 = P[k + 1], p3 = P[Math.min(P.length - 1, k + 2)];
        const n = Math.ceil((Math.hypot(p2[0] - p1[0], p2[1] - p1[1]) * 3) / S.UX);
        for (let q = 0; q <= n; q++) {
          const t = q / n, t2 = t * t, t3 = t2 * t;
          const f = (a0, a1, a2, a3) => 0.5 * (2 * a1 + (a2 - a0) * t + (2 * a0 - 5 * a1 + 4 * a2 - a3) * t2 + (3 * a1 - a0 - 3 * a2 + a3) * t3);
          const i = Math.floor(S.BX0 + f(p0[0], p1[0], p2[0], p3[0]) / S.UX), j = Math.floor(S.BY0 + f(p0[1], p1[1], p2[1], p3[1]) / S.UY);
          const c = cells[cells.length - 1];
          if (!c || c[0] !== i || c[1] !== j) cells.push([i, j]);
        }
      }
      return cells;
    }
    // start a signal on tract e leaving node `from`; it runs on through up to 3-5 tracts, never turning by more than
    // 40 degrees (one tract is shorter than the trail). `prev` is the arriving point when it fires on from a signal.
    function fire(e, from, prev) {
      let cur = other(e, from);
      const first = wiggle(from, cur, prev);
      if (!first) return false;                 // (fire-on only) it would turn sharply
      const P = [NODES[from], ...first, NODES[cur]], hops = 3 + Math.floor(Math.random() * 3);
      let edge = e;
      for (let h = 1; h < hops; h++) {
        const opts = [];
        for (const q of ADJ[cur]) {
          if (q === edge) continue;
          const nx = other(q, cur), w = wiggle(cur, nx, P[P.length - 2]);
          if (w) opts.push([q, nx, w]);
        }
        if (!opts.length) break;
        const [q, nx, w] = opts[Math.floor(Math.random() * opts.length)];
        P.push(...w, NODES[nx]); edge = q; cur = nx;
      }
      addSignal(P, edge, cur);
      return true;
    }
    function addSignal(P, edge, to, delay = 0) {   // polyline P (unit coords) -> a signal; each lasts < 1 s
      const v = rnd(50, 65) / S.UX, tl = rnd(12, 16) / S.UX, cells = splineCells(P), ease = delay > 0;
      const keep = Math.max(4, Math.floor((ease ? 0.72 : 0.9) * v - tl));   // head travel + trail drain (+ ease-in) < 1 s
      const cut = cells.length > keep || to < 0;
      signals.push({ cells: cells.length > keep ? cells.slice(0, keep) : cells, e: edge, to: cut ? -1 : to, last: P[P.length - 2], s: 0, v, tl, age: -delay, ease });
    }
        function inBrain(x, y) { const i = Math.floor(S.BX0 + x / S.UX), j = Math.floor(S.BY0 + y / S.UY); return !!(S.MASK[j] && S.MASK[j][i]); }
    // an object falling in: 3-4 trails start where it hit and run off in all directions (evenly spread around the
    // circle, jittered), each bending at most ~23 degrees per 4-unit step. The brain is a 3-D object seen from one
    // side: a trail that reaches the outline runs on two steps past it and so slides off the edge (cells outside the
    // outline are not drawn) instead of turning back.
    function impactTrails(ci, cj) {
      const x0 = (ci + 0.5 - S.BX0) * S.UX, y0 = (cj + 0.5 - S.BY0) * S.UY;
      const n = 3 + (Math.random() < 0.5 ? 1 : 0), th0 = rnd(0, Math.PI * 2);
      for (let k = 0; k < n; k++) {
        let th = th0 + (k * Math.PI * 2) / n + rnd(-0.3, 0.3), x = x0, y = y0, out = -1;
        const P = [[x, y]];
        for (let st = 0; st < 10 && out < 2; st++) {
          th += rnd(-0.4, 0.4); x += Math.cos(th) * 4; y += Math.sin(th) * 4; P.push([x, y]);
          if (out >= 0 || !inBrain(x, y)) out++;
        }
        addSignal(P, -1, -1, 0.001 + 0.05 * k + rnd(0, 0.03));   // staggered launch
      }
    }
    function sigCap() { return Math.max(1, Math.round((1 + progress + (phase === 'in' ? 1.5 : 0)) * Math.sqrt(S.k))); }
    function stepSignals(dt) {
      if (signals.length < sigCap() && Math.random() < dt * (2 + progress + (phase === 'in' ? 1.5 : 0))) {   // capped by sigCap: usually one at a time
        const e = Math.floor(Math.random() * TRACTS.length); fire(e, TRACTS[e][Math.random() < 0.5 ? 0 : 1]);
      }
      for (const g of signals) {
        g.age = (g.age || 0) + dt;
        if (g.age < 0) continue;                         // impact trails launch staggered
        const e = g.ease ? Math.min(1, g.age / 0.15) : 1;   // ...and ease in: slow and dim, then full speed
        g.s += g.v * dt * (g.ease ? 0.3 + 0.7 * e * e * (3 - 2 * e) : 1);
        if (!g.done && g.s >= g.cells.length - 1) {       // arrived: the region lights briefly, and may fire on
          g.done = true;
          const [i, j] = g.cells[g.cells.length - 1];
          if (g.to < 0) continue;                       // trimmed mid-route: it just fades out (no glow, no fire-on)
          pulses.push({ i, j, r: 0, life: 0.3, max: 0.3, hit: false });
          const next = ADJ[g.to].filter((q) => q !== g.e);
          let n = Math.random() < 0.25 ? 1 : 0; if (Math.random() < 0.05) n++;
          for (let q = 0; q < n && next.length && signals.length < sigCap() + 1; q++) fire(next.splice(Math.floor(Math.random() * next.length), 1)[0], g.to, g.last);   // refused if it would turn sharply
        }
      }
      signals = signals.filter((g) => g.s - g.tl < g.cells.length - 1);   // gone once the trail has drained in
    }

    function step(dt) {
      t += dt;
      if (finaleReq && (phase === 'orbit' || phase === 'reform')) {
        finaleReq = false; big = true; phase = 'in'; cycStart = t - tO;
        const add = Math.max(0, 16 - objs.length); for (let i = 0; i < add; i++) objs.push(makeObj(i, add, 1));
        objs.forEach((o) => { o.fade = 1; o.r = Math.min(o.r, 1); });
      }
      const ct = t - cycStart;
      if (phase === 'orbit' && ct > tO) phase = 'in';
      if (phase === 'in' && ct > tO + tI) {
        phase = 'burst'; flashUntil = t + (big ? 1.3 : 0.7);
        burst(Math.round((big ? 180 : 70 + 50 * progress) * S.k), big ? 1.6 : 1.15);
      }
      if (phase === 'burst' && ct > tO + tI + 0.45 * TS) { phase = 'reform'; spawn(1.6); }
      if (phase === 'reform' && ct > tC) { phase = 'orbit'; cycStart = t; big = false; }
      const speedUp = 1 + 0.6 * progress;
      for (const o of objs) {
        let w = (o.w * speedUp) / TS;
        if (phase === 'in') {
          const p = clamp((ct - tO) / tI, 0, 1);
          o.r = Math.min(o.r, 1 - p * p * (1.15 - 0.15 * p));           // accelerate inwards
          w *= 1 + 9 * p * p;
        } else if (phase === 'reform' || o.r > 1) {
          o.r += (1 - o.r) * Math.min(1, (dt * 2.2) / TS); o.fade = Math.min(1, o.fade + (dt * 1.4) / TS);
        } else { o.fade = 1; }
        o.th += w * dt;
        if (Math.random() < (dt * 3) / TS) o.fr++;
        const P = pos(o);
        // perspective: depth from the orbit phase (front half = near); 3 levels with hysteresis (no flicker)
        const zn = Math.sin(o.th), hy = 0.1;
        if (o.lod < 2 && zn > 0.3 + hy) o.lod = 2; else if (o.lod === 2 && zn < 0.3 - hy) o.lod = 1;
        if (o.lod > 0 && zn < -0.35 - hy) o.lod = 0; else if (o.lod === 0 && zn > -0.35 + hy) o.lod = 1;
        if (phase === 'in') { o.trail.push(P); if (o.trail.length > 7) o.trail.shift(); } else if (o.trail.length) o.trail.shift();
        if (phase === 'in' && o.alive && o.r < 0.36) {                    // swallowed: trails run off from where it hit
          o.alive = false;
          let best = S.EDGE[0], bd = 1e9;
          for (const e of S.EDGE) { const dd = (e.i - P.x) ** 2 + ((e.j - P.y) * S.aspect) ** 2; if (dd < bd) { bd = dd; best = e; } }
          const oi = Math.round(P.x), oj = Math.round(P.y);              // start where it vanished if that is on the brain
          if (S.MASK[oj] && S.MASK[oj][oi]) impactTrails(oi, oj); else impactTrails(best.i, best.j);
        }
      }
      stepSignals(dt);                       // neural activity: signals traveling between regions
      for (const p of pulses) p.life -= dt;
      pulses = pulses.filter((p) => p.life > 0);
      for (const s of sparks) { s.life -= dt; s.x += s.vx * dt; s.y += s.vy * dt; s.vx *= 0.93; s.vy *= 0.93; s.vy += dt * 1.5; }
      sparks = sparks.filter((s) => s.life > 0 && s.x > -2 && s.x < S.W + 2 && s.y > -2 && s.y < S.H + 2);
    }

    function render() {
      const { W, H, MASK, aspect } = S;
      const ch = Array.from({ length: H }, () => Array(W).fill(' '));
      const cl = Array.from({ length: H }, () => Array(W).fill(''));
      const put = (x, y, rows, c, front, opaque) => {   // opaque: blank the sprite's own row spans (readable over the brain)
        const y0 = Math.round(y - (rows.length - 1) / 2);
        for (let r = 0; r < rows.length; r++) {
          const yy = y0 + r; if (yy < 0 || yy >= H) continue;
          const s = rows[r], x0 = Math.round(x - s.length / 2);
          const lo = opaque ? s.search(/\S/) : 0, hi = opaque ? s.trimEnd().length : 0;
          for (let q = 0; q < s.length; q++) {
            const xx = x0 + q;
            if (xx < 0 || xx >= W || (!front && MASK[yy][xx])) continue;
            if (s[q] === ' ') { if (opaque && q > lo && q < hi) { ch[yy][xx] = ' '; cl[yy][xx] = ''; } continue; }
            ch[yy][xx] = s[q]; cl[yy][xx] = c;
          }
        }
      };
      for (const s of S.STARS) {                                         // faint twinkling background
        if (!MASK[s.y][s.x] && Math.sin(t * 1.7 + s.ph) > 0.55) { ch[s.y][s.x] = s.ph % 3 < 1 ? '·' : '.'; cl[s.y][s.x] = 'bg'; }
      }
      const items = objs.map((o) => ({ o, P: pos(o) }));
      const drawTrail = (o, front) => { o.trail.forEach((p, q) => put(p.x, p.y, [q > 3 ? '·' : '.'], 'tr-' + o.kd.k, front)); };
      const drawObj = ({ o, P }, front) => {
        if (!o.alive) return;
        if (o.fade < 1 && Math.random() > o.fade) { put(P.x, P.y, ['·'], o.kd.k, front); return; }
        const fr = o.kd.lv[LODS[o.lod]];
        put(P.x, P.y, fr[o.fr % fr.length], o.lod === 0 ? o.kd.k + ' far' : o.kd.k, front, front && o.lod === 2 && LODS[2] === 3);
      };
      for (const it of items) if (it.P.z <= 0) { drawTrail(it.o, false); drawObj(it, false); }
      // the brain: slow breathing, a light wave, neural pulses; brighter as things fall in; flash on the burst
      const flash = t < flashUntil, ff = flash ? (flashUntil - t) / (big ? 1.3 : 0.7) : 0;
      const breathe = 0.06 * Math.sin(t * 2.1);
      const inP = phase === 'in' ? clamp((t - cycStart - tO) / tI, 0, 1) : 0;
      const HOT = new Float32Array(W * H);                               // activity glow (0..1) per cell
      const disc = (ci, cj, R, v) => {                                   // a soft dot of radius R columns
        for (let j = Math.max(0, Math.floor(cj - R / aspect)); j <= Math.min(H - 1, Math.ceil(cj + R / aspect)); j++)
          for (let i = Math.max(0, Math.floor(ci - R)); i <= Math.min(W - 1, Math.ceil(ci + R)); i++) {
            const a = v * Math.max(0, 1 - Math.hypot(i - ci, (j - cj) * aspect) / (R + 0.5));
            if (a > HOT[j * W + i]) HOT[j * W + i] = a;
          }
      };
      for (const p of pulses) {
        disc(p.i, p.j, 0.8 + 0.5 * S.k, 0.8 * (p.life / p.max));      // arrival: a brief glow, no growth
      }
      for (const g of signals) {                                         // head (a dot) and its fading trail
        if (g.age < 0) continue;
        const n = g.cells.length - 1, hd = Math.min(n, g.s), a = g.ease ? Math.min(1, g.age / 0.12) : 1;
        for (let q = Math.max(0, Math.ceil(g.s - g.tl)); q <= hd; q++) {
          const f = 1 - (g.s - q) / g.tl, [i, j] = g.cells[q];
          if (!MASK[j] || !MASK[j][i]) continue;                         // stays inside the outline
          if (f * 0.84 * a > HOT[j * W + i]) HOT[j * W + i] = f * 0.84 * a;   // one cell wide: a thin line fading linearly
        }
        if (g.s <= n) { const [i, j] = g.cells[Math.floor(hd)]; if (MASK[j] && MASK[j][i]) HOT[j * W + i] = Math.max(HOT[j * W + i], a); }   // head: one bright cell, no blob
      }
      for (const c of S.BRAIN) {
        let d = c.d * (1 + breathe + 0.07 * Math.sin(t * 1.3 - c.x * 0.06 + c.y * 0.05)) + 0.25 * inP * inP;
        const hot = HOT[c.j * W + c.i];
        let cls = d < 0.5 ? 'b1' : d < 0.85 ? 'b2' : 'b3', gap = '';
        if (hot > 0.04) {                                                // activity: head > trail, fading in five steps
          cls = hot > 0.85 ? 'sh' : hot > 0.6 ? 'hot' : hot > 0.38 ? 'sm' : hot > 0.18 ? 'sg' : 'sf';
          if (c.d > 0.2) d += 0.4 * hot; else gap = hot > 0.6 ? '+' : hot > 0.35 ? ':' : '·';   // crossing a sulcus
        }
        if (flash) { d += 0.6 * ff; cls = c.d > 0.2 ? 'flash' : 'flash2'; }
        const idx = c.d < 0.02 && !flash ? 0 : clamp(Math.round(clamp(d, 0, 1) * (RAMP.length - 1)), 1, RAMP.length - 1);
        let g = (flash ? RAMP : BRAMP)[idx];
        if (c.part === 'cb' && !flash && idx) g = d > 0.5 ? '=' : '-';
        if (c.part === 'st' && !flash && idx) g = d > 0.45 ? '|' : ':';
        if (gap && !flash) { ch[c.j][c.i] = gap; cl[c.j][c.i] = cls; continue; }
        ch[c.j][c.i] = g; cl[c.j][c.i] = idx ? cls : '';
      }
      for (const it of items) if (it.P.z > 0) { drawTrail(it.o, true); drawObj(it, true); }
      for (const s of sparks) {
        const f = s.life / s.max, sp = Math.hypot(s.vx, s.vy * aspect);
        let g;
        if (f > 0.55 && sp > 9) { const a = Math.atan2(s.vy * aspect, s.vx); g = '-\\|/'[((Math.round(a / (Math.PI / 4)) % 4) + 4) % 4]; }
        else g = f > 0.45 ? '*' : f > 0.22 ? '+' : f > 0.1 ? '·' : '.';
        put(s.x, s.y, [g], s.cls, true);
      }
      let html = '';
      for (let y = 0; y < H; y++) {
        let cur = '', run = '';
        for (let x = 0; x < W; x++) {
          if (cl[y][x] !== cur) { if (run) html += cur ? `<span class="mm-${cur.replace(' ', ' mm-')}">${run}</span>` : run; cur = cl[y][x]; run = ''; }
          run += esc(ch[y][x]);
        }
        if (run) html += cur ? `<span class="mm-${cur.replace(' ', ' mm-')}">${run}</span>` : run;
        if (y < H - 1) html += '\n';
      }
      pre.innerHTML = html;
    }

    function frame(now) {
      if (stopped) return;
      raf = requestAnimationFrame(frame);
      if (now - last < (opts.fit ? 66 : 50)) return;                     // ~20 fps (15 full screen)
      const dt = last ? Math.min(0.1, (now - last) / 1000) : 0.05; last = now;
      if (document.hidden || !pre.offsetParent) return;                  // tab or element hidden: skip the work
      ensureScene(); step(dt); render();
    }

    pre.setAttribute('aria-hidden', 'true');
    if (reduce) {                                                         // one still frame: brain + orbiters
      const still = () => { if (stopped) return; if (!pre.offsetParent) { raf = setTimeout(still, 500); return; }
        progress = Math.max(progress, 0.5); ensureScene(); for (let q = 0; q < 25; q++) stepSignals(0.05); render(); };
      still();
      return { setProgress() {}, finale() {}, stop() { stopped = true; clearTimeout(raf); } };
    }
    raf = requestAnimationFrame(frame);
    return {
      setProgress(p) { progress = clamp(+p || 0, 0, 1); },
      finale() { finaleReq = true; },
      stop() { stopped = true; cancelAnimationFrame(raf); },
      _debug: {                                                       // for screenshot scripts: jump ahead in time
        seek(sec) { ensureScene(); for (let q = 0; q < sec * 20; q++) step(0.05); render(); return phase; } },
    };
  }
  window.Mindmeld = { mount };
})();

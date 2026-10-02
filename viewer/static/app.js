// The viewer's front end: draws the scorer's and the visualizer's outputs in 3D and as videos.
// Every array comes from the offline stages; nothing here recomputes a metric.
//
// Paths and clouds carry no object identity: a point's colour is a scalar of its own path.
// The Objects view and index.mp4 colour by object id, fixed by the id alone.
//
// All panels share one WebGL canvas behind the grid; each 3D panel renders into its own
// scissor rectangle with its own scene, camera and orbit controls. One context serves every
// panel because browsers cap the number of WebGL contexts per page.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';

const TINT = { gt: 0x6f8ba6, pred: 0xe0883c };
const PLAIN = new THREE.Color(0xa9b4c0);
const SKIN = new THREE.Color(0.83, 0.86, 0.9);
// Canvas clear colours matching the page tokens --surface (between panels) and
// --viewport (inside a panel).
const GAP_INK = 0xffffff;
const VIEWPORT_INK = 0xf2f4f4;
const VIEWS = ['masks', 'index', 'scene', 'trajectory', 'meshes', 'objects', 'overlap',
               'defects', 'depth', 'flow', 'tracks', 'features'];
const VIEW_NAMES = ['Masks', 'Index', 'Scene', 'Trajectory', 'Meshes', 'Objects', 'Overlap',
                    'Defects', 'Depth', 'Flow', 'Tracks', 'Features'];
// Views drawn from one frame of `meshes/`, fetched per frame. `meshes` draws the soup plain,
// `objects` colours it by object id, and `overlap` and `defects` paint red the components
// that interpenetration and the mesh checks flag.
const GEOMETRY = { meshes: true, objects: true, overlap: true, defects: true };
// COLOURED: views that use the colour and trail controls. THINNED: views that use the
// density control.
const COLOURED = { trajectory: true };
const THINNED = { scene: true, trajectory: true };
const FEATURE_MODELS = [['dinov3', 'DINOv3'], ['tips', 'TIPSv2']];
const MEDIA_COLUMNS = 3;
// Display names of the two sides, used by every tag, header, checkbox and legend.
const SIDES = ['gt', 'pred'];
const SIDE_NAMES = { gt: 'GT', pred: 'Reconstruction' };
// Media views, as rows of [title, cells]; a cell is `[file, caption, side, name]`. A
// `source:` file is the case video or the render, served in place. `side` sets the cell's
// tag and frame colour; a `both` cell carries the name of its picture in `name`.
const MEDIA = {
  masks: [['the take', [['source:reference', 'the case video', 'gt'],
                        ['source:render', 'the delivered render', 'pred']]],
          ['dynamic masks — a body is dynamic when its motion points travel more than alpha of its own size',
           [['reference_masks.mp4', 'under its dynamic mask', 'gt'],
            ['render_masks.mp4', 'under its dynamic mask', 'pred'],
            ['masks_overlap.mp4', 'blue — GT only · orange — reconstruction only · white — both',
             'both', 'Overlap']]]],
  index: [['the take', [['source:render', 'the delivered render', 'pred']]],
          ['per-pixel object ids, rasterised from meshes/ through camera.json by the scorer — a colour is fixed by the id, dark is empty background',
           [['index.mp4', 'object ids', 'pred'], ['index_overlay.mp4', 'the ids over the render', 'pred']]]],
  depth: [['the take', [['source:reference', 'the case video', 'gt'],
                        ['source:render', 'the delivered render', 'pred']]],
          ['depth, normalised per frame', [['reference_depth.mp4', 'Video Depth Anything on the case video', 'gt'],
                                           ['render_depth.mp4', 'rasterised from the submitted meshes', 'pred'],
                                           ['depth_error.mp4', 'over the pixels the rasteriser covers',
                                            'both', 'Error']]]],
  flow: [['the take', [['source:reference', 'the case video', 'gt'],
                       ['source:render', 'the delivered render', 'pred']]],
         ['optical flow on RAFT’s colour wheel — hue is direction, saturation is speed, saturating at 2% of the image diagonal; dark grey where a side does not answer',
          [['reference_flow.mp4', 'RAFT on the reference video', 'gt'],
           ['render_flow.mp4', 'read off the submitted geometry', 'pred'],
           ['flow_error.mp4', 'endpoint error in radii over the pixels read, black outside',
            'both', 'Error']]]],
  tracks: [['the take', [['source:reference', 'reference video', 'gt'],
                         ['source:render', 'render', 'pred']]],
           ['screen paths — a dot and its last frames, one colour per query row. Both draw the same rows, the reference’s movers Track2D reads, so a query is path i on both',
            [['reference_tracks.mp4', 'CoTracker3 on the reference video', 'gt'],
             ['render_tracks.mp4', 'followed off the submitted geometry', 'pred'],
             ['matches.mp4', 'both sides on the reference video, every pair joined',
              'both', 'Paired']]]],
  features: [['the take', [['source:reference', 'reference video', 'gt'],
                           ['source:render', 'render', 'pred']]],
             ...FEATURE_MODELS.map(([model, title]) => [`${title} patch PCA, one basis for both`,
               [[`reference_${model}.mp4`, '', 'gt'], [`render_${model}.mp4`, '', 'pred']]])],
};
const COLOURS = ['age', 'height', 'displacement'];
const COLOUR_NAMES = ['Trail age', 'Height', 'Displacement'];
const SPACE_NAMES = { original: 'Original', registered: 'Registered' };
const FAR = 1e9;   // an aFrame no scrub can reach: the segment is never drawn
const HELD = 32;   // frames of geometry kept in the browser, per kind

const state = {
  world: null, case: null, entry: null, manifest: null, pair: null, sides: ['pred'],
  panels: ['trajectory'], space: 'original',
  frame: 0, playing: false, trail: 24, colour: 'age', fps: 24, density: 100,
  overlay: false,
  show: { gt: true, pred: true },
  wanted: null, drawn: null,
};

// ---------------------------------------------------------------- scene setup

const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio);
stage.appendChild(renderer.domElement);

function resize() {
  // updateStyle on, so the canvas gets a CSS size and not just a backing store
  renderer.setSize(Math.max(1, stage.clientWidth), Math.max(1, stage.clientHeight));
}
addEventListener('resize', resize);

// -------------------------------------------------------------- array access

// Google's Turbo colour map, as nine control points.
const TURBO = [[0.190, 0.072, 0.232], [0.246, 0.401, 0.869], [0.128, 0.735, 0.938],
               [0.145, 0.940, 0.609], [0.581, 0.999, 0.242], [0.930, 0.816, 0.184],
               [0.988, 0.520, 0.128], [0.868, 0.196, 0.056], [0.480, 0.016, 0.011]];

// The same control points as a 9x1 texture, linearly filtered, for the shader.
const TURBO_TEXTURE = (() => {
  const data = new Uint8Array(TURBO.length * 4);
  TURBO.forEach((c, i) => {
    data[i * 4] = c[0] * 255; data[i * 4 + 1] = c[1] * 255;
    data[i * 4 + 2] = c[2] * 255; data[i * 4 + 3] = 255;
  });
  const texture = new THREE.DataTexture(data, TURBO.length, 1, THREE.RGBAFormat);
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
})();

function turbo(t, into) {
  const x = Math.max(0, Math.min(1, t)) * (TURBO.length - 1);
  const i = Math.min(TURBO.length - 2, Math.floor(x)), f = x - i;
  return into.setRGB(TURBO[i][0] + f * (TURBO[i + 1][0] - TURBO[i][0]),
                     TURBO[i][1] + f * (TURBO[i + 1][1] - TURBO[i][1]),
                     TURBO[i][2] + f * (TURBO[i + 1][2] - TURBO[i][2]));
}

const KINDS = { int8: Int8Array, uint8: Uint8Array, uint16: Uint16Array, int32: Int32Array,
                float32: Float32Array };

// A manifest plus its blob: one for the world's points, one per frame of geometry.
function pack(manifest, blob) {
  return {
    manifest, blob,
    array(name) {
      const spec = manifest.arrays[name];
      if (!spec) return null;
      const count = spec.shape.reduce((a, b) => a * b, 1);
      return { data: new (KINDS[spec.dtype])(blob, spec.offset, count),
               shape: spec.shape };
    },
  };
}

function array(name) {
  return state.pair ? state.pair.array(name) : null;
}

// The spaces the bundle offers: a pair ships registered, a lone world in its own metres.
function spaces() {
  return state.manifest && state.manifest.pair ? ['registered', 'original'] : ['original'];
}

// The bundle's per-side matrix from world metres into the registered frame; identity in
// the original space.
function intoShared(side, space) {
  const matrix = new THREE.Matrix4();
  if (space === 'original' || !state.manifest.placement) return matrix;
  return matrix.set(...state.manifest.placement[side]);
}

// The group matrix for a side's bundle arrays in `space`.
function placement(side, space) {
  const matrix = intoShared(side, 'registered');
  return space === 'original' ? matrix.invert() : matrix.identity();
}

function unit() {
  return state.space === 'original' ? 'm' : 'radii';
}

// ------------------------------------------------------------------- geometry

// One LineSegments per side. A segment joins frame i to i+1 and both vertices carry i+1,
// so the shader culls by frame and trail. A segment touching a frame outside the point's
// life carries FAR.
const PATH_VERTEX = `
  attribute float aFrame;
  attribute vec3 aColour;
  uniform float uMax;
  uniform float uTrail;
  uniform float uAgeRamp;
  uniform sampler2D uTurbo;
  varying vec3 vColour;

  // a texture, not a uniform array: GLSL ES 1.00 forbids a runtime index
  vec3 turbo(float t) {
    return texture2D(uTurbo, vec2(clamp(t, 0.0, 1.0), 0.5)).rgb;
  }

  void main() {
    if (aFrame > uMax || aFrame <= uMax - uTrail) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return;
    }
    if (uAgeRamp > 0.5) {
      float age = clamp((uMax - aFrame) / max(uTrail, 1.0), 0.0, 1.0);
      vColour = turbo(0.88 - 0.80 * age) * (1.0 - 0.55 * age);
    } else {
      vColour = aColour;
    }
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;
const PATH_FRAGMENT = `
  varying vec3 vColour;
  void main() { gl_FragColor = vec4(vColour, 1.0); }`;

// The rows the density slider keeps, spread evenly over all rows.
function subset(count) {
  const wanted = Math.max(1, Math.round(count * state.density / 100));
  const rows = new Int32Array(wanted);
  for (let i = 0; i < wanted; i++) {
    rows[i] = Math.min(count - 1, Math.round((i * count) / wanted));
  }
  return rows;
}

// The offset of a point's first finite frame, or -1. It stands in for the missing frames
// so the bounding box stays finite; the segments that use it are culled by frame number.
function anchorOf(paths, p, frames) {
  for (let f = 0; f < frames; f++) {
    const at = (p * frames + f) * 3;
    if (Number.isFinite(paths.data[at])) return at;
  }
  return -1;
}

function pathObject(paths, colourOf, offset = 0) {
  const [count, frames] = paths.shape;
  const rows = subset(count);
  const segments = rows.length * (frames - 1) * 2;
  const position = new Float32Array(segments * 3);
  const colour = new Float32Array(segments * 3);
  const frame = new Float32Array(segments);
  let w = 0;
  for (const p of rows) {
    const fallback = anchorOf(paths, p, frames);
    if (fallback < 0) { w += (frames - 1) * 2; continue; }
    for (let f = 0; f < frames - 1; f++) {
      const a = (p * frames + f) * 3, b = a + 3;
      const live = Number.isFinite(paths.data[a]) && Number.isFinite(paths.data[b]);
      const c = colourOf(p, f + 1);
      for (const src of [a, b]) {
        const from = live ? src : fallback;
        position[w * 3] = paths.data[from];
        position[w * 3 + 1] = paths.data[from + 1];
        position[w * 3 + 2] = paths.data[from + 2];
        colour[w * 3] = c.r; colour[w * 3 + 1] = c.g; colour[w * 3 + 2] = c.b;
        frame[w] = live ? offset + f + 1 : FAR;
        w++;
      }
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('aColour', new THREE.BufferAttribute(colour, 3));
  geometry.setAttribute('aFrame', new THREE.BufferAttribute(frame, 1));
  const material = new THREE.ShaderMaterial({
    vertexShader: PATH_VERTEX, fragmentShader: PATH_FRAGMENT,
    uniforms: {
      uMax: { value: 1e9 }, uTrail: { value: 1e9 }, uAgeRamp: { value: 0 },
      uTurbo: { value: TURBO_TEXTURE },
    },
    transparent: false,
  });
  return new THREE.LineSegments(geometry, material);
}

// A per-point colour from its path on Turbo: start height, or start-to-end displacement.
function stableColour(paths, mode) {
  const [count, frames] = paths.shape;
  const value = new Float32Array(count);
  let low = Infinity, high = -Infinity;
  for (let p = 0; p < count; p++) {
    const start = anchorOf(paths, p, frames);
    if (start < 0) { value[p] = NaN; continue; }
    if (mode === 'height') {
      value[p] = paths.data[start + 2];
    } else {
      let last = start;
      for (let f = frames - 1; f >= 0; f--) {
        const at = (p * frames + f) * 3;
        if (Number.isFinite(paths.data[at])) { last = at; break; }
      }
      const dx = paths.data[last] - paths.data[start];
      const dy = paths.data[last + 1] - paths.data[start + 1];
      const dz = paths.data[last + 2] - paths.data[start + 2];
      value[p] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    if (value[p] < low) low = value[p];
    if (value[p] > high) high = value[p];
  }
  const span = (high - low) || 1;
  const scratch = new THREE.Color();
  return (p) => (Number.isFinite(value[p])
    ? turbo((value[p] - low) / span, scratch) : PLAIN);
}

// The path heads, rebuilt each frame from the points alive on it.
function headObject(paths, colour, stable, offset = 0) {
  const [count, frames] = paths.shape;
  const rows = subset(count);
  const position = new Float32Array(rows.length * 3);
  const tint = new Float32Array(rows.length * 3);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(tint, 3));
  const points = new THREE.Points(geometry, new THREE.PointsMaterial({
    color: stable ? 0xffffff : colour, vertexColors: !!stable,
    size: 3, sizeAttenuation: false,
  }));
  points.userData.move = (f) => {
    const at = Math.min(Math.max(f - offset, 0), frames - 1);
    let w = 0;
    for (const p of rows) {
      const src = (p * frames + at) * 3;
      if (!Number.isFinite(paths.data[src])) continue;
      position[w * 3] = paths.data[src];
      position[w * 3 + 1] = paths.data[src + 1];
      position[w * 3 + 2] = paths.data[src + 2];
      if (stable) {
        const c = stable(p);
        tint[w * 3] = c.r; tint[w * 3 + 1] = c.g; tint[w * 3 + 2] = c.b;
      }
      w++;
    }
    geometry.setDrawRange(0, w);
    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate = true;
  };
  return points;
}

function cloudObject(name, colour) {
  const points = array(name);
  if (!points) return null;
  const rows = subset(points.shape[0]);
  const position = new Float32Array(rows.length * 3);
  rows.forEach((p, i) => {
    position[i * 3] = points.data[p * 3];
    position[i * 3 + 1] = points.data[p * 3 + 1];
    position[i * 3 + 2] = points.data[p * 3 + 2];
  });
  return new THREE.Points(
    new THREE.BufferGeometry().setAttribute(
      'position', new THREE.BufferAttribute(position, 3)),
    new THREE.PointsMaterial({ color: colour, size: 2, sizeAttenuation: false }));
}

// ------------------------------------------------------------------- panels

// One panel per view on screen. A 3D panel owns a scene, camera and orbit controls and
// renders into the shared canvas; a media panel owns its videos and draws over the canvas.
let panels = [];

function makePanel(view) {
  const element = document.createElement('section');
  element.className = `panel ${MEDIA[view] ? 'media' : 'gl'}`;
  const header = document.createElement('header');
  const name = document.createElement('span');
  name.className = 'name';
  name.textContent = VIEW_NAMES[VIEWS.indexOf(view)];
  const sides = document.createElement('span');
  sides.className = 'sides';
  const drop = document.createElement('button');
  drop.textContent = '×';
  drop.title = 'drop this panel';
  drop.onclick = () => togglePanel(view, false);
  header.append(name, sides, drop);
  const body = document.createElement('div');
  body.className = 'body';
  element.append(header, body);
  const panel = { view, element, body, sides, videos: [] };
  if (!MEDIA[view]) setUpGl(panel);
  return panel;
}

function setUpGl(panel) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(VIEWPORT_INK);
  scene.add(new THREE.AmbientLight(0xffffff, 1.6));
  const key = new THREE.DirectionalLight(0xffffff, 1.4);
  key.position.set(1, -1, 2);
  scene.add(key);
  const floor = new THREE.GridHelper(10, 10, 0xc3ccd6, 0xdde3ea);
  floor.rotation.x = Math.PI / 2;               // the helper is Y-up; lay it flat
  scene.add(floor);
  const content = new THREE.Group();
  scene.add(content);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 10000);
  camera.up.set(0, 0, 1);                       // the contract's world space is Z-up
  // the controls listen on the panel body because the shared canvas lies behind every panel
  const controls = new OrbitControls(camera, panel.body);
  controls.enableDamping = true;
  const legend = document.createElement('div');
  legend.className = 'legend';
  const ruler = document.createElement('div');
  ruler.className = 'ruler';
  ruler.innerHTML = '<div class="bar"></div><span></span>';
  panel.body.append(legend, ruler);
  Object.assign(panel, { scene, camera, controls, content, floor, legend, ruler,
                         built: null, overlay: null, framedAt: null });
}

function disposePanel(panel) {
  if (panel.controls) panel.controls.dispose();
  if (panel.content) clear(panel.content);
  for (const video of panel.videos) { video.pause(); video.removeAttribute('src'); }
  panel.videos.length = 0;
}

// Rebuild the grid from state.panels, keeping surviving panels and their cameras.
function syncPanels() {
  const host = document.getElementById('grid');
  const kept = new Map(panels.map((panel) => [panel.view, panel]));
  const next = state.panels.map((view) => kept.get(view) || makePanel(view));
  for (const panel of panels) if (!next.includes(panel)) disposePanel(panel);
  panels = next;
  host.replaceChildren(...panels.map((panel) => panel.element));
  host.dataset.count = panels.length <= 6 ? String(panels.length) : 'many';
}

// Set every camera's aspect from its panel before any panel is built, so the first framing
// uses the panel's aspect.
function sizeCameras() {
  for (const panel of panels) {
    if (!panel.camera) continue;
    const box = panel.body.getBoundingClientRect();
    const aspect = (box.width || 1) / (box.height || 1);
    if (panel.camera.aspect === aspect) continue;
    panel.camera.aspect = aspect;
    panel.camera.updateProjectionMatrix();
  }
}

// A video framed in its side's colour, with the side's name on a solid chip in the top-left
// corner so the name stays legible over any frame.
function shot(video, side, name) {
  const box = document.createElement('div');
  box.className = `shot framed ${side}`;
  const tag = document.createElement('span');
  tag.className = `side-tag ${side}`;
  tag.textContent = name || SIDE_NAMES[side];
  box.append(tag, video);
  return box;
}

// ---------------------------------------------------------------- view builds

function clear(group) {
  if (!group) return;
  group.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    if (node.material) node.material.dispose();
  });
  group.clear();
}

// Build a points view (scene, trajectory) from the bundle arrays, which hold the whole take;
// a scrub only updates what is drawn. A geometry panel is cleared here and filled by
// `drawGeometry`.
function buildPoints(panel) {
  clear(panel.content);
  panel.built = { heads: [], paths: [] };
  panel.overlay = null;
  if (!state.manifest) return;
  if (GEOMETRY[panel.view]) return;

  for (const side of state.sides) {
    if (!state.show[side]) continue;
    const colour = TINT[side];
    const group = new THREE.Group();
    group.matrixAutoUpdate = false;
    group.matrix.copy(placement(side, state.space));

    if (panel.view === 'scene') {
      const cloud = cloudObject(`scene_${side}`, colour);
      if (cloud) group.add(cloud);
    } else if (panel.view === 'trajectory') {
      const paths = array(`path_${side}`);
      if (paths && paths.shape[0]) {
        const age = state.colour === 'age';
        const stable = age ? null : stableColour(paths, state.colour);
        const flat = new THREE.Color(colour);
        const line = pathObject(paths, stable ? (p) => stable(p) : () => flat);
        line.material.uniforms.uAgeRamp.value = age ? 1 : 0;
        group.add(line); panel.built.paths.push(line);
        const head = headObject(paths, colour, stable);
        group.add(head); panel.built.heads.push(head);
      }
    }
    panel.content.add(group);
  }
  // a lone world's MoGe estimate, placed in the world's metres by the bundler
  if (panel.view === 'scene' && state.show.gt) {
    const estimate = cloudObject('scene_moge', TINT.gt);
    if (estimate) {
      const group = new THREE.Group();
      group.matrixAutoUpdate = false;
      group.matrix.copy(placement('pred', state.space));
      group.add(estimate);
      panel.content.add(group);
    }
  }
  // the group for the trajectory's mesh overlay, inside `content` so the framing offset
  // applies to it
  if (panel.view === 'trajectory' && state.overlay) {
    panel.overlay = new THREE.Group();
    panel.content.add(panel.overlay);
  }
  panel.dirty = true;
  applyFrame(panel);
  reframe(panel);
  legend(panel);
}

// Frame the camera once per world, space and view kind; view switches, toggles and scrubs
// keep the camera where it is.
function reframe(panel) {
  const stamp = `${state.world}|${state.space}|`
    + `${GEOMETRY[panel.view] ? 'geometry' : panel.view}`;
  if (stamp === panel.framedAt) return;
  panel.framedAt = stamp;
  frameCamera(panel);
}

// ------------------------------------------------------- one frame of geometry

const held = new Map();          // `${kind}|${world}|${frame}` -> pack

// One asynchronous draw at a time. `want` names what to draw. While a draw is in flight a
// newer `want` is recorded; when the draw lands, `showFrame` runs again if `want` changed.
// A stale result is never drawn over a newer view.
let inflight = false;
const note = (text) => { document.getElementById('note').textContent = text; };

async function settle(want, draw) {
  if (state.drawn === want) return;
  state.wanted = want;
  if (inflight) return;
  inflight = true;
  note('reading the frame…');
  try {
    await draw();
    note('');
  } catch (error) {
    note(String(error));
  } finally {
    inflight = false;
  }
  if (state.wanted !== want) showFrame();
}

async function fetchFrame(kind, world, frame) {
  const name = `${kind}|${world}|${frame}`;
  if (held.has(name)) return held.get(name);
  const manifest = await (await fetch(`/${kind}/${world}/${frame}`)).json();
  if (manifest.error) throw new Error(manifest.error);
  const blob = await (await fetch(`/${kind}blob/${world}/${frame}`)).arrayBuffer();
  const item = pack(manifest, blob);
  held.set(name, item);
  for (const stale of [...held.keys()].slice(0, Math.max(0, held.size - HELD))) {
    held.delete(stale);
  }
  return item;
}

// Panels that need a frame of `meshes/`: the geometry views, and a trajectory panel with
// the mesh overlay on. One fetch per side serves all of them.
function geometryPanels() {
  return panels.filter((panel) => GEOMETRY[panel.view]
    || (panel.view === 'trajectory' && state.overlay));
}

// Draw one frame of mesh into every panel that needs it. A frame that lands after the scrub
// moved on is kept in `held` and not drawn.
async function drawGeometry(frame) {
  const wanted = geometryPanels();
  if (!wanted.length || !state.manifest) return;
  const want = `${wanted.map((panel) => panel.view).join(',')}|${state.world}|${frame}|`
    + `${state.density}|${state.space}|${JSON.stringify(state.show)}`;
  const sides = state.sides.filter((side) => state.show[side]);
  const worlds = sides.map((side) => (side === 'gt' ? state.manifest.gt_world : state.world));
  await settle(want, async () => {
    const pieces = [];
    for (const [i, side] of sides.entries()) {
      if (!worlds[i]) continue;
      pieces.push({ side, geometry: await fetchFrame('mesh', worlds[i], frame) });
    }
    // the scrub moved on; the fetched frame stays in `held`
    if (state.wanted !== want) return;
    for (const panel of wanted) {
      if (GEOMETRY[panel.view]) {
        clear(panel.content);
        panel.built = { heads: [], paths: [] };
        for (const piece of pieces) buildSoup(panel, panel.content, piece, false);
        reframe(panel);
      } else if (panel.overlay) {
        clear(panel.overlay);
        for (const piece of pieces) buildSoup(panel, panel.overlay, piece, true);
      }
      panel.dirty = true;
      legend(panel);
    }
    state.drawn = want;
  });
}

// Fault colour: on Overlap a component's red scales with its buried fraction; on Defects a
// component that failed a mesh check is red.
const FAULT = new THREE.Color(0.88, 0.19, 0.19);

// One colour per object id for the Objects view, keyed by the world's id so an object keeps
// its colour across frames. Hue steps by the golden angle.
const OBJECT_INK = new Map();

function objectInk(id) {
  let ink = OBJECT_INK.get(id);
  if (!ink) {
    ink = new THREE.Color().setHSL((id * 0.618033988749895) % 1, 0.62, 0.58);
    OBJECT_INK.set(id, ink);
  }
  return ink;
}

// `ghost` draws the soup under the trajectory's paths: one translucent colour, no depth
// write, and nothing counted for the legend.
function buildSoup(panel, target, { side, geometry }, ghost) {
  const verts = geometry.array('verts');
  const faces = geometry.array('faces');
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  group.matrix.copy(intoShared(side, state.space));
  if (!verts || !faces || !verts.shape[0]) { target.add(group); return; }

  const count = verts.shape[0];
  const part = geometry.array('part').data;
  const buried = geometry.array('buried').data;
  const clean = geometry.array('clean').data;
  const colour = new Float32Array(count * 3);
  const tint = new THREE.Color();
  const fault = (k) => (panel.view === 'overlap' ? (buried[k] > 0 ? 0.35 + 0.65 * Math.min(1, buried[k] / 0.5) : 0)
    : panel.view === 'defects' ? (clean[k] === 0 ? 1 : 0) : 0);
  const owner = geometry.array('owner').data;
  const seen = new Set();
  for (let v = 0; v < count; v++) {
    const slot = part[v];
    if (!ghost && panel.view === 'objects') {
      const id = owner[slot];
      seen.add(id);
      tint.copy(objectInk(id));
    } else if (ghost) {
      tint.copy(SKIN);
    } else {
      tint.copy(SKIN).lerp(FAULT, fault(slot));
    }
    colour[v * 3] = tint.r; colour[v * 3 + 1] = tint.g; colour[v * 3 + 2] = tint.b;
  }
  if (!ghost) {
    const built = panel.built;
    built.objects = (built.objects || 0) + seen.size;
    built.components = (built.components || 0) + geometry.manifest.components;
    built.buried = (built.buried || 0) + buried.reduce((n, f) => n + (f > 0), 0);
    built.failing = (built.failing || 0) + clean.reduce((n, c) => n + (c === 0), 0);
  }
  const mesh = new THREE.BufferGeometry();
  mesh.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts.data), 3));
  mesh.setAttribute('color', new THREE.BufferAttribute(colour, 3));
  mesh.setIndex(new THREE.BufferAttribute(new Uint32Array(faces.data), 1));
  mesh.computeVertexNormals();
  group.add(new THREE.Mesh(mesh, new THREE.MeshLambertMaterial({
    vertexColors: true, side: THREE.DoubleSide,
    transparent: ghost, opacity: ghost ? 0.42 : 1, depthWrite: !ghost,
  })));
  target.add(group);
}

// ------------------------------------------------------ moving the playhead

// Move the playhead: seek the videos, update the points panels, draw the geometry frame.
function showFrame() {
  if (!state.manifest) return;
  seekMedia();
  for (const panel of panels) applyFrame(panel);
  frameText();
  markPlot();
  drawGeometry(state.frame);
}

function applyFrame(panel) {
  const built = panel.built;
  if (!built) return;
  for (const line of built.paths || []) {
    line.material.uniforms.uMax.value = state.frame;
    line.material.uniforms.uTrail.value = state.trail;
  }
  for (const head of built.heads || []) head.userData.move(state.frame);
}

let frameLabel = null;
function frameText() {
  const scrub = document.getElementById('scrub');
  const text = `frame ${state.frame} / ${scrub.max}`;
  if (text === frameLabel) return;
  frameLabel = text;
  document.getElementById('frame-text').textContent = text;
}

// ------------------------------------------------------------------ framing

function drawnBox(panel) {
  panel.content.position.set(0, 0, 0);
  panel.content.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(panel.content);
  return box.isEmpty() || !Number.isFinite(box.min.x) ? null : box;
}

// Translate the drawn content so its centre is the origin, then fit the camera to it.
function frameCamera(panel) {
  const box = drawnBox(panel);
  if (!box) return;
  const camera = panel.camera;
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const centre = box.getCenter(new THREE.Vector3());
  panel.content.position.copy(centre).negate();
  panel.content.updateMatrixWorld(true);
  panel.floor.scale.setScalar(size / 10);
  panel.floor.position.set(0, 0, box.min.z - centre.z);
  panel.dirty = true;

  // back off until the box fits the narrower field-of-view angle, looking at the origin
  const vertical = camera.fov * Math.PI / 360;
  const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
  const angle = Math.max(0.05, Math.min(vertical, horizontal));
  const reach = (size / 2) / Math.tan(angle) * 1.15;
  const direction = new THREE.Vector3(0.8, -0.9, 0.5).normalize();
  camera.position.set(0, 0, 0).addScaledVector(direction, reach);
  panel.controls.target.set(0, 0, 0);
  clip(panel);
  panel.controls.update();
}

const clipBox = new THREE.Box3();

// Set the near and far planes from the panel's bounding sphere, which is recomputed only
// when the content is marked dirty.
function clip(panel) {
  const camera = panel.camera;
  if (panel.dirty || !panel.sphere) {
    clipBox.setFromObject(panel.scene);
    panel.dirty = false;
    panel.sphere = clipBox.isEmpty() ? null
      : clipBox.getBoundingSphere(panel.sphere || new THREE.Sphere());
  }
  if (!panel.sphere) return;
  const radius = Math.max(panel.sphere.radius, 1e-6);
  const distance = camera.position.distanceTo(panel.sphere.center);
  const far = (distance + radius) * 1.05;
  const near = Math.max((distance - radius) * 0.5, far / 1e4, 1e-6);
  if (near === camera.near && far === camera.far) return;
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();
}

// A screen-space ruler: how long 120 px is at the distance the camera sits.
function ruler(panel, height) {
  const camera = panel.camera;
  const distance = camera.position.distanceTo(panel.controls.target);
  const perPixel = 2 * distance * Math.tan(camera.fov * Math.PI / 360) / Math.max(1, height);
  const raw = perPixel * 120;
  if (!Number.isFinite(raw) || raw <= 0) return;
  const power = Math.pow(10, Math.floor(Math.log10(raw)));
  const nice = [1, 2, 5, 10].find((m) => m * power >= raw) * power;
  const digits = nice < 1 ? 3 : nice < 10 ? 1 : 0;
  const reading = `${nice / perPixel}px|${nice.toFixed(digits)} ${unit()}`;
  if (panel.rulerAt === reading) return;      // a still camera writes nothing
  panel.rulerAt = reading;
  const [width, text] = reading.split('|');
  panel.ruler.firstChild.style.width = width;
  panel.ruler.lastChild.textContent = text;
}

// A side's name in the legend, as a chip in that side's colour.
const sideKey = (side) => `<b class="side ${side}">${SIDE_NAMES[side]}</b>`;

function legend(panel) {
  const m = state.manifest;
  if (!m || !panel.legend) return;
  const bits = [];
  if (m.pair && state.space === 'original') {
    bits.push(`the reconstruction is ${(1 / m.alignment.scale).toFixed(2)}x the GT`);
  }
  if (panel.view === 'scene' && m.scene) {
    bits.push('frame 0: every pixel of that frame’s rasterised depth,'
      + ' back-projected through the world’s own camera');
    for (const side of state.sides) {
      const info = m.scene.sides[side];
      if (info) bits.push(`${sideKey(side)} ${info.points} of ${info.of} covered pixels`);
    }
    const moge = m.scene.sides.moge;
    if (moge) {
      bits.push(`${sideKey('gt')} estimate, ${moge.points} of ${moge.of} valid points — the MoGe-3`
        + ' estimate of frame 0, unit to unit in the world’s own metres');
    }
  }
  if (panel.view === 'trajectory' && m.trajectory) {
    const t = m.trajectory;
    bits.push(m.pair
      ? 'every sample the metric drew, from its birth to its end, where the reference camera sees it'
      : 'every sample the viewer drew, from its birth to its end, where the world camera sees it');
    for (const side of state.sides) {
      const rows = t.rows[side];
      if (rows !== undefined) bits.push(`${sideKey(side)} ${rows} paths`);
    }
    const error = t.error.dtw;
    if (error !== null && error !== undefined) {
      bits.push(`DTW error ${error.toFixed(4)} (mean matched gap, in reference radii, capped at 1)`);
    }
    if (state.overlay) bits.push('the delivered soup of this frame, under the tracks');
  }
  if (GEOMETRY[panel.view]) {
    const b = panel.built || {};
    if (panel.view === 'meshes') bits.push(`${b.components || 0} components, the soup as delivered`);
    if (panel.view === 'objects') bits.push(`${b.objects || 0} objects, each in its own ink by the id it carries in the world`);
    if (panel.view === 'overlap') bits.push(`${b.components || 0} components · red: buried in another solid, ${b.buried || 0} of them`);
    if (panel.view === 'defects') bits.push(`${b.components || 0} components · red: failed a mesh check, ${b.failing || 0} of them`);
    for (const [name, item] of held) {
      const [kind, world, frame] = name.split('|');
      if (kind !== 'mesh' || +frame !== state.frame) continue;
      bits.push(`<b>${world}</b>, frame ${frame}: ${item.manifest.vertices} vertices, `
        + `${item.manifest.triangles} triangles`);
    }
  }
  const html = bits.join('<br>');
  if (panel.legendAt === html) return;
  panel.legendAt = html;
  panel.legend.innerHTML = html;
}

// ---------------------------------------------------------------------- panel

function buttons(id, names, get, set) {
  const host = document.getElementById(id);
  host.innerHTML = '';
  names.forEach((name, index) => {
    const button = document.createElement('button');
    button.textContent = name;
    button.className = get() === index ? 'on' : '';
    button.onclick = () => { set(index); buttons(id, names, get, set); refresh(); };
    host.appendChild(button);
  });
}

function slider(id, value, set, label) {
  const input = document.getElementById(id);
  const text = document.getElementById(`${id}-text`);
  input.value = value;
  text.textContent = label(value);
  input.oninput = () => {
    text.textContent = label(+input.value);
    set(+input.value);
  };
}

// A row whose value is null renders as a group heading.
function table(id, rows) {
  document.getElementById(id).innerHTML = rows.map(
    ([k, v]) => (v === null ? `<tr class="head"><td colspan="2">${k}</td></tr>`
      : `<tr><td>${k}</td><td>${v}</td></tr>`)).join('');
}

const fmt = (v) => (v === null || v === undefined ? '—' : (+v).toFixed(3));

const scored = new Map();   // world -> what /metrics answered for it

// The reward table's groups. A reward key not listed here is shown under `Other`; a listed
// key absent from a world's reward shows an em-dash.
const REWARD_GROUPS = [
  ['Read against the reference world', [
    ['dynamic IoU', 'dynamic_iou'], ['scene 3D', 'scene_3d'], ['trajectory DTW', 'trajectory_dtw'],
    ['EMD step', 'emd_step'], ['occupancy DTW', 'occupancy_dtw'], ['uni3D point', 'uni3d_point_scene']]],
  ['Read against the video', [
    ['flow', 'flow_distribution'], ['track2D', 'track2d_dtw'], ['uni3D MoGe', 'uni3d_moge_scene'],
    ['geoPhys DINOv3', 'geophys_dinov3']]],
  ['World quality', [
    ['interpenetration', 'interpenetration'], ['mesh watertight', 'mesh_watertight'],
    ['mesh manifold', 'mesh_manifold'], ['mesh clean faces', 'mesh_clean_faces'],
    ['mesh no self-intersection', 'mesh_no_self_intersection']]],
];
const REWARD_KNOWN = new Set(REWARD_GROUPS.flatMap(([, keys]) => keys.map(([, key]) => key)));
// Rows for every listed reward, with an em-dash where the world has none.
function rewardRows(s) {
  const rows = [];
  for (const [title, keys] of REWARD_GROUPS) {
    rows.push([title, null]);
    for (const [label, key] of keys) rows.push([label, fmt(s[key])]);
  }
  // VIDEO_KEYS appear in the video table, so they are left out of `Other`. It is read at
  // call time because it is declared below.
  const elsewhere = new Set(VIDEO_KEYS.map(([, key]) => key));
  const rest = Object.keys(s).filter(
    (key) => !REWARD_KNOWN.has(key) && !elsewhere.has(key)).sort();
  if (rest.length) {
    rows.push(['Other', null]);
    for (const key of rest) rows.push([key.replace(/_/g, ' '), fmt(s[key])]);
  }
  return rows;
}

// The reward table. A world with no reward, such as a case without a reference world,
// hides the block.
function scoreTable() {
  const found = scored.get(state.world);
  const block = document.getElementById('score-block');
  const note = document.getElementById('score-note');
  note.className = '';
  if (found === undefined) { block.className = ''; table('scores', [['reading…', '—']]); return; }
  if (found.error) {
    block.className = '';
    table('scores', [['unavailable', '—']]);
    note.textContent = found.error;
    note.className = 'bad';
    return;
  }
  if (found.status === 'gate_failure') {
    block.className = '';
    table('scores', [['gate failure', '—']]);
    note.textContent = (found.timeline || []).join('; ') || 'the timeline gate rejected this world';
    note.className = 'bad';
    return;
  }
  if (!found.reward) { block.className = 'gone'; return; }
  block.className = '';
  table('scores', rewardRows(found.reward));
  note.textContent = 'results/reward.json';
}

// Fetch /metrics once per world; it feeds both score tables.
async function fetchMetrics(world) {
  if (scored.has(world)) return;
  let payload;
  try {
    payload = await (await fetch(`/metrics/${world}`)).json();
  } catch (error) {
    payload = { error: String(error) };
  }
  scored.set(world, payload);
  if (state.world !== world) return;
  scoreTable();
  videoTable();
}

// ---------------------------------------------------- read against the video

// Semantic and Depth are read against the case video and shown in their own table.
const VIDEO_KEYS = [['semantic (DINOv3)', 'semantic_dinov3'],
                    ['semantic (TIPS)', 'semantic_tips'],
                    ['depth error', 'depth_error']];

function videoTable() {
  const found = scored.get(state.world) || {};
  const block = document.getElementById('video-block');
  const note = document.getElementById('video-note');
  note.className = '';
  if (!found.video || !VIDEO_KEYS.some(([, k]) => found.video[k] !== undefined)) {
    block.className = 'gone';
    scoreFold();
    return;
  }
  block.className = '';
  table('video-scores', VIDEO_KEYS.map(([label, k]) => [label, fmt(found.video[k])]));
  note.textContent = 'results/reward.detail.json, read against the case reference video';
  scoreFold();
}

// Hide the score disclosure when both tables are hidden. Runs after both are drawn.
function scoreFold() {
  const empty = document.getElementById('score-block').className === 'gone'
    && document.getElementById('video-block').className === 'gone';
  document.getElementById('score-fold').className = empty ? 'gone' : '';
}

// The media files /worlds listed for the world on screen.
function builtMedia() {
  return new Set((state.entry && state.entry.media) || []);
}

function mediaUrl(file) {
  return file.startsWith('source:')
    ? `/source/${state.world}/${file.slice('source:'.length)}.mp4`
    : `/media/${state.world}/${file}`;
}

// A media view's rows, keeping the cells whose file exists; `source:` cells are always kept.
function mediaRows(view) {
  const have = builtMedia();
  return MEDIA[view]
    .map(([title, cells]) => [title, cells.filter(([file]) => file.startsWith('source:') || have.has(file))])
    .filter(([, cells]) => cells.length);
}

// A media view is offered when it has a rendered file besides the source videos.
function viewAvailable(view) {
  if (MEDIA[view]) return mediaRows(view).length > 1;   // more than the take alone
  return true;
}

// The videos have no controls: the scrub bar and play button drive them, and a frame maps
// to a fraction of each video's duration. During play the videos run on their own, the
// playhead follows the first, and the others are resynced when they drift.
const media = [];
function seekMedia() {
  const frames = +document.getElementById('scrub').max + 1;
  for (const video of media) {
    if (!video.duration) continue;
    const at = Math.min(video.duration, state.frame * video.duration / frames);
    if (Math.abs(video.currentTime - at) > 1e-3) video.currentTime = at;
  }
}

function playMedia(scrub) {
  const [lead, ...rest] = media;
  if (!lead || !lead.duration) return;
  const frames = +scrub.max + 1;
  if (!state.playing) {
    for (const video of media) if (!video.paused) video.pause();
    return;
  }
  const rate = state.fps * lead.duration / frames;
  for (const video of media) {
    if (video.playbackRate !== rate) video.playbackRate = rate;
    if (video.paused) video.play();
  }
  for (const video of rest) {
    if (Math.abs(video.currentTime - lead.currentTime) > 0.08) video.currentTime = lead.currentTime;
  }
  state.frame = Math.min(frames - 1, Math.floor(lead.currentTime / lead.duration * frames));
  scrub.value = state.frame;
}

function drawMedia(panel) {
  panel.body.innerHTML = '';
  panel.videos.length = 0;
  // each row has one column per cell, up to MEDIA_COLUMNS
  for (const [title, cells] of mediaRows(panel.view)) {
    const row = document.createElement('div');
    row.className = 'media-row';
    row.innerHTML = `<p class="head">${title}</p>`;
    const grid = document.createElement('div');
    grid.className = 'media-grid';
    grid.style.gridTemplateColumns = `repeat(${Math.min(cells.length, MEDIA_COLUMNS)}, 1fr)`;
    for (const [file, label, side, name] of cells) {
      const figure = document.createElement('figure');
      const video = document.createElement('video');
      video.muted = video.playsInline = video.loop = true;
      video.preload = 'auto';
      video.src = mediaUrl(file);
      video.onloadedmetadata = seekMedia;
      figure.append(shot(video, side, name));
      if (label) {
        const caption = document.createElement('figcaption');
        caption.textContent = label;
        figure.append(caption);
      }
      grid.append(figure);
      panel.videos.push(video);
    }
    row.append(grid);
    panel.body.append(row);
  }
}

// ---------------------------------------------------------------- the series plot
//
// The /series plot. A reading has one of two shapes: `frame` is plotted against time
// under the playhead, and `path` as a sorted cost curve. The curve is drawn when the world or the reading changes; each frame
// moves only the playhead line and the value readout.
const PLOT_W = 1000, PLOT_H = 300, PLOT_PAD = 10;
const SHAPE_NAMES = { frame: 'through time', path: 'per path' };

const plot = { key: null, at: null, series: null, marked: null, said: null };
const seriesOf = new Map();     // world -> what /series answered for it

async function fetchSeries(world) {
  if (seriesOf.has(world)) return;
  let payload;
  try {
    payload = await (await fetch(`/series/${world}`)).json();
  } catch (error) {
    payload = { error: String(error), series: [] };
  }
  seriesOf.set(world, payload);
  if (state.world === world) drawPlot();
}

// Fill the reading picker, grouped by shape, and return the selected reading.
function plotPick() {
  const found = seriesOf.get(state.world);
  const rows = (found && found.series) || [];
  const pick = document.getElementById('plot-pick');
  const stamp = rows.map((row) => row.key).join(',');
  if (pick.dataset.at !== stamp) {
    pick.dataset.at = stamp;
    pick.innerHTML = Object.entries(SHAPE_NAMES).map(([shape, title]) => {
      const held = rows.filter((row) => row.shape === shape);
      return held.length ? `<optgroup label="${title}">` + held.map(
        (row) => `<option value="${row.key}">${row.label}</option>`).join('') + '</optgroup>' : '';
    }).join('');
  }
  // keep the chosen reading across worlds when the new world has it
  if (!rows.some((row) => row.key === plot.key)) {
    const first = rows.find((row) => row.shape === 'frame') || rows[0];
    plot.key = first ? first.key : null;
  }
  pick.value = plot.key || '';
  pick.disabled = !rows.length;
  return rows.find((row) => row.key === plot.key) || null;
}

// An SVG path through the values, broken at every null or NaN.
function polyline(values, low, span) {
  const n = values.length;
  const at = (i) => (n < 2 ? PLOT_W / 2 : (i / (n - 1)) * PLOT_W);
  const parts = [];
  let open = false;
  values.forEach((value, i) => {
    if (value === null || !Number.isFinite(value)) { open = false; return; }
    const y = PLOT_H - PLOT_PAD - ((value - low) / span) * (PLOT_H - 2 * PLOT_PAD);
    parts.push(`${open ? 'L' : 'M'}${at(i).toFixed(1)} ${y.toFixed(1)}`);
    open = true;
  });
  return parts.join(' ');
}

const finiteOf = (values) => values.filter((v) => v !== null && Number.isFinite(v));

function drawPlot() {
  const host = document.getElementById('plot');
  const found = seriesOf.get(state.world);
  const row = plotPick();
  // three states: a reading, not fetched yet, and fetched with no readings
  const stamp = `${state.world}|${row ? row.key : (found ? 'empty' : 'pending')}`;
  if (plot.at === stamp) return;
  plot.at = stamp;
  plot.series = row;
  plot.marked = plot.said = null;
  host.classList.toggle('has', !!row);
  const svg = host.querySelector('svg');
  const path = (name) => svg.querySelector(`.curve.${name}`);
  for (const name of ['gt', 'pred', 'one']) path(name).setAttribute('d', '');
  const text = (name, value) => { host.querySelector(`.${name}`).textContent = value; };
  if (!row) {
    host.querySelector('.none').textContent = !found ? 'reading…'
      : found.error ? String(found.error)
      : 'nothing per-frame was written for this world';
    return;
  }
  const lines = [['one', row.values]];
  const all = lines.flatMap(([, values]) => finiteOf(values || []));
  const low = all.length ? Math.min(...all) : 0;
  const high = all.length ? Math.max(...all) : 1;
  const span = (high - low) || 1;
  for (const [name, values] of lines) {
    if (values && values.length) path(name).setAttribute('d', polyline(values, low, span));
  }
  text('title', row.label);
  text('hi', all.length ? high.toPrecision(3) : '');
  text('lo', all.length ? low.toPrecision(3) : '');
  host.querySelector('.keys').innerHTML = '';
  if (row.shape === 'path') {
    text('x0', 'cheapest');
    text('x1', `${row.paths} paths, sorted`);
  } else {
    text('x0', 'frame 0');
    text('x1', `frame ${((row.values.length - 1) * (row.stride || 1))}`);
  }
  svg.querySelector('.playhead').hidden = row.shape !== 'frame';
  markPlot();
}

// Move the playhead line and the value readout; the rest of the plot is unchanged.
function markPlot() {
  const row = plot.series;
  if (!row || !take.open) return;
  const host = document.getElementById('plot');
  if (row.shape !== 'frame') {
    if (plot.said !== '') { plot.said = ''; host.querySelector('.read').textContent = ''; }
    return;
  }
  const n = row.values.length;
  const i = Math.max(0, Math.min(n - 1, Math.round(state.frame / (row.stride || 1))));
  if (i !== plot.marked) {
    plot.marked = i;
    const x = (n < 2 ? PLOT_W / 2 : (i / (n - 1)) * PLOT_W).toFixed(1);
    const line = host.querySelector('.playhead');
    line.setAttribute('x1', x);
    line.setAttribute('x2', x);
  }
  const value = row.values[i];
  const said = value === null || !Number.isFinite(value) ? '—' : (+value).toPrecision(4);
  if (said !== plot.said) { plot.said = said; host.querySelector('.read').textContent = said; }
}

// -------------------------------------------------------------- the source videos
//
// The case video and the render, shown above the grid whatever panels are picked. Their
// elements join the same `media` list as the panels' videos, so one playhead drives all.
const TAKE_SIDES = [['reference', 'gt'], ['render', 'pred']];
const take = { open: true, at: null, videos: [] };

// Rebuilt only when the world or the fold state changes, so panel toggles and slider moves
// do not reload the videos.
function syncTake() {
  const stamp = `${state.world}|${take.open}`;
  if (take.at === stamp) return;
  take.at = stamp;
  const host = document.getElementById('take-pair');
  document.getElementById('take').classList.toggle('shut', !take.open);
  document.getElementById('take-toggle').setAttribute('aria-expanded', String(take.open));
  // folding removes the sources, which stops the fetch and frees the decoders
  for (const video of take.videos) { video.pause(); video.removeAttribute('src'); video.load(); }
  take.videos.length = 0;
  host.replaceChildren();
  if (!take.open || !state.world) return;
  for (const [kind, side] of TAKE_SIDES) {
    const figure = document.createElement('figure');
    const video = document.createElement('video');
    video.muted = video.playsInline = video.loop = true;
    video.preload = 'auto';
    video.src = `/source/${state.world}/${kind}.mp4`;
    video.onloadedmetadata = seekMedia;
    // a missing video (a real case without reference.mp4, or a world without a render)
    // returns 404; the element is replaced by a notice
    video.onerror = () => {
      for (const list of [media, take.videos]) {
        const at = list.indexOf(video);
        if (at >= 0) list.splice(at, 1);
      }
      video.remove();
      const said = document.createElement('div');
      said.className = `absent ${side}`;
      said.textContent = `no ${SIDE_NAMES[side]} video for this world`;
      figure.replaceChildren(said);
    };
    figure.append(shot(video, side));
    host.append(figure);
    take.videos.push(video);
  }
}

// Rebuild the list every seek and play step walks. The source videos come first, so the
// first of them leads playback.
function remedia() {
  media.length = 0;
  for (const video of take.videos) media.push(video);
  for (const panel of panels) for (const video of panel.videos) media.push(video);
  seekMedia();
}

// The sides a panel draws: for a media view the sides of its cells, for a 3D view the
// world's sides that the Show boxes allow.
function panelSides(panel) {
  if (!MEDIA[panel.view]) return state.sides.filter((side) => state.show[side]);
  const seen = new Set(mediaRows(panel.view).flatMap(
    ([, cells]) => cells.map(([, , side]) => side)));
  return SIDES.filter((side) => seen.has(side) || seen.has('both'));
}

// Name the side in the panel header when the panel shows exactly one side.
function paintSides(panel) {
  const sides = panelSides(panel);
  const stamp = sides.length === 1 ? sides[0] : '';
  if (panel.sides.dataset.at === stamp) return;
  panel.sides.dataset.at = stamp;
  if (!stamp) { panel.sides.replaceChildren(); return; }
  const chip = document.createElement('b');
  chip.className = stamp;
  chip.textContent = SIDE_NAMES[stamp];
  panel.sides.replaceChildren(chip);
}

// The side panel: score tables, alignment and world info, and the controls. Each control
// block is shown only while a panel on screen uses it.
function sidePanel() {
  const m = state.manifest;
  scoreTable();
  videoTable();

  const align = document.getElementById('align-block');
  align.className = m.pair ? '' : 'gone';
  if (m.pair) {
    const a = m.alignment;
    table('align', [
      ['scale', a.scale.toFixed(4)],
      ['rotation', `${a.rotation_deg.toFixed(2)}°`],
      ['residual', a.residual.toFixed(4)],
      ['reference radius', `${a.radius.toFixed(3)} m`]]);
  }

  const side = state.sides[state.sides.length - 1];
  table('info', [
    ['frames', m.frames[side]],
    ['resolution', m.resolution[side].join(' x ')]]);

  const hasMoge = !!(m.scene && m.scene.sides && m.scene.sides.moge);
  document.getElementById('show-block').className = (m.pair || hasMoge) ? '' : 'gone';
  const shown = (kinds) => (state.panels.some((view) => kinds[view]) ? '' : 'gone');
  document.getElementById('density-block').className = shown(THINNED);
  document.getElementById('trail-block').className = shown(COLOURED);
  document.getElementById('colour-block').className = shown(COLOURED);
  document.getElementById('overlay-block').className = shown({ trajectory: true });
}

// ------------------------------------------------------------------ load

let loading = 0;      // the newest load; an older one that lands later stops

async function load(world) {
  const ticket = ++loading;
  note('reading…');
  const manifest = await (await fetch(`/media/${world}/bundle.json`)).json();
  if (manifest.error) { note(manifest.error); return; }
  const blob = await (await fetch(`/media/${world}/bundle.bin`)).arrayBuffer();
  if (ticket !== loading) return;
  state.manifest = manifest;
  state.sides = manifest.sides;
  state.pair = pack(manifest, blob);
  state.world = world;
  state.entry = allWorlds.find((row) => row.world === world) || null;
  if (state.entry) state.case = state.entry.case;
  state.drawn = null;
  held.clear();
  for (const panel of panels) { clear(panel.content); panel.built = null; }
  note('');
  drawPickers();

  // a pair opens registered unless a space was chosen; a lone world has only its own metres
  if (!spaces().includes(state.space) || (manifest.pair && !state.spaceChosen)) state.space = spaces()[0];
  const frames = Math.min(...Object.values(manifest.frames));
  const scrub = document.getElementById('scrub');
  scrub.max = frames - 1;
  state.frame = Math.min(state.frame, frames - 1);
  scrub.value = state.frame;
  for (const panel of panels) panel.framedAt = null;
  drawViews();
  drawSpaces();
  await refresh();
  fetchMetrics(world);
  fetchSeries(world);
}

async function refresh() {
  if (!state.manifest) return;
  syncPanels();
  sizeCameras();
  sidePanel();
  syncTake();
  for (const panel of panels) {
    // a media panel's videos are rebuilt only when the world changes
    if (!MEDIA[panel.view]) buildPoints(panel);
    else if (panel.mediaAt !== state.world) { drawMedia(panel); panel.mediaAt = state.world; }
    paintSides(panel);
  }
  remedia();
  drawPlot();
  state.drawn = null;       // a points panel is built in place; nothing is pending
  frameText();
  await drawGeometry(state.frame);
}

// The view buttons. Only views this world can show are offered; picked panels that are not
// offered are dropped, falling back to the first offered view.
function drawViews() {
  const offered = VIEWS.filter(viewAvailable);
  state.panels = state.panels.filter((view) => offered.includes(view));
  if (!state.panels.length) state.panels = [offered[0]];
  const host = document.getElementById('views');
  host.innerHTML = '';
  for (const view of offered) {
    const button = document.createElement('button');
    button.textContent = VIEW_NAMES[VIEWS.indexOf(view)];
    button.className = state.panels.includes(view) ? 'on' : '';
    button.onclick = () => togglePanel(view, !state.panels.includes(view));
    host.appendChild(button);
  }
}

// The last panel cannot be dropped.
function togglePanel(view, on) {
  if (!on && state.panels.length === 1) return;
  state.panels = VIEWS.filter(
    (other) => (other === view ? on : state.panels.includes(other)));
  drawViews();
  refresh();
}

function drawSpaces() {
  const names = spaces();
  buttons('spaces', names.map((s) => SPACE_NAMES[s]),
    () => names.indexOf(state.space), (i) => { state.space = names[i]; state.spaceChosen = true; });
}

// The two pickers: the case first, then a world of that case (the reference or a run).
const modelName = (row) => (row.reference ? 'GT — the reference world' : row.slug);

function drawPickers() {
  const scenes = document.getElementById('scene');
  const worlds = document.getElementById('world');
  const cases = [...new Set(allWorlds.map((row) => row.case))];
  scenes.innerHTML = cases.map((name) => `<option value="${name}">${name}</option>`).join('');
  scenes.value = state.case;
  const models = allWorlds.filter((row) => row.case === state.case);
  worlds.innerHTML = models.map(
    (row) => `<option value="${row.world}">${modelName(row)}</option>`).join('');
  worlds.value = state.world;
}


// ------------------------------------------------------------------ run table

// The run table: one row per model × effort, averaged over cases. The server returns each
// run's record and rewards; the grouping and means are computed here.
//
// A run is executable when the agent exited cleanly, delivered a world and its scorer status
// is not `gate_failure`. Rewards are averaged over executable runs and spend over all runs.
// Trials of a case are averaged first, so every case has equal weight.
//
// Tables: world quality, main metrics, additional metrics and spend; an arrow marks the
// better direction. The registration residual (trimmed two-way nearest-neighbour distance,
// in reference radii) is shown at the fit's start and end. The reference scored against
// itself is the GT split and is kept out of the others.
const BOARD_TABLES = [
  ['World quality', [['interpenetration ↑', 'interpenetration', 3],
    ['watertight ↑', 'mesh_watertight', 3], ['manifold ↑', 'mesh_manifold', 3],
    ['clean faces ↑', 'mesh_clean_faces', 3], ['no self-intersection ↑', 'mesh_no_self_intersection', 3]]],
  ['Main metrics', [
    ['DINOv3 ↑', 'semantic_dinov3', 3], ['flow ↑', 'flow_distribution', 3], ['track2D ↑', 'track2d_dtw', 3],
    ['dynamic IoU ↑', 'dynamic_iou', 3], ['depth error ↓', 'depth_error', 3], ['scene 3D ↑', 'scene_3d', 3],
    ['trajectory DTW ↑', 'trajectory_dtw', 3], ['EMD step ↑', 'emd_step', 3]]],
  ['Additional metrics', [['registration residual ↓', 'residual', 3],
    ['TIPS ↑', 'semantic_tips', 3], ['geoPhys DINOv3 ↑', 'geophys_dinov3', 3],
    ['uni3D MoGe ↑', 'uni3d_moge_scene', 3], ['uni3D point ↑', 'uni3d_point_scene', 3],
    ['occupancy DTW ↑', 'occupancy_dtw', 3]]],
  ['Spend per task', [['time (min) ↓', 'time', 1], ['tokens (M) ↓', 'tokens', 2],
    ['cost ($) ↓', 'cost', 2], ['steps ↓', 'steps', 0], ['tool calls ↓', 'tool_calls', 0]]],
];
const BOARD_WIDTH = 8;      // readings a table holds before the rest go under it
const BOARD_REWARDS = BOARD_TABLES.slice(0, 3).flatMap(([, columns]) => columns.map(([, key]) => key));
const BOARD_SPENT = {
  time: (r) => (r.agent_seconds === null || r.agent_seconds === undefined ? null : r.agent_seconds / 60),
  tokens: (r) => r.usage && (r.usage.input_tokens + r.usage.cached_input_tokens
    + (r.usage.cache_write_tokens || 0) + r.usage.output_tokens) / 1e6,
  cost: (r) => r.cost_usd, steps: (r) => r.steps, tool_calls: (r) => r.tool_calls,
};
// Case coverage differs between groups, so every average carries its case count. `basis`
// selects whether a group is averaged over all its cases or only the cases every shown group
// shares; `only` is the set of shown groups, null for all.
const board = { open: false, runs: null, split: 'all', rows: 'average',
                basis: 'all', counts: true, only: null };

const executable = (r) => r.agent_exit === 0 && !!r.delivered && r.status !== 'gate_failure';
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const finite = (xs) => xs.filter((x) => x !== null && x !== undefined && Number.isFinite(x));
// residual and residual_start share one case set
const basisKey = (key) => (key === 'residual_start' ? 'residual' : key);
const AVERAGED = ['residual', ...BOARD_REWARDS, ...Object.keys(BOARD_SPENT)];

// Per-case summary of one group's runs: executability over all trials, rewards over the
// executable ones, spend over all. `n` holds the number of runs behind each mean.
function caseSummary(runs) {
  const good = runs.filter(executable);
  const row = { runs: runs.length, executable: good.length, cases: 1,
                executability: good.length / runs.length, n: { runs: runs.length } };
  const take = (key, values) => {
    const kept = finite(values);
    row[key] = mean(kept);
    row.n[key] = kept.length;
  };
  for (const key of BOARD_REWARDS) take(key, good.map((r) => r.reward && r.reward[key]));
  take('residual', good.map((r) => r.residual));
  take('residual_start', good.map((r) => r.residual_start));
  for (const [key, read] of Object.entries(BOARD_SPENT)) take(key, runs.map(read));
  return row;
}

// The mean of per-case summaries. `allow` filters the cases per reading; `n` holds the case
// count behind each mean.
function groupSummary(perCase, allow) {
  const of = (key) => perCase.filter(([name]) => !allow || allow(basisKey(key), name))
    .map(([, c]) => c);
  const base = of('runs');
  const row = { runs: base.reduce((a, c) => a + c.runs, 0),
                executable: base.reduce((a, c) => a + c.executable, 0),
                cases: base.length, n: { runs: base.length, executability: base.length } };
  row.executability = mean(base.map((c) => c.executability));
  for (const key of [...AVERAGED, 'residual_start']) {
    const kept = finite(of(key).map((c) => c[key]));
    row[key] = mean(kept);
    row.n[key] = kept.length;
  }
  return row;
}

// For each reading, the cases where every shown group has a value.
function sharedCases(shown) {
  const allowed = new Map();
  for (const key of ['runs', ...AVERAGED]) {
    let kept = null;
    for (const [, perCase] of shown) {
      const has = new Set(perCase.filter(([, c]) => (key === 'runs' ? c.runs > 0
        : Number.isFinite(c[key]))).map(([name]) => name));
      kept = kept === null ? has : new Set([...kept].filter((name) => has.has(name)));
    }
    allowed.set(key, kept || new Set());
  }
  return (key, name) => (allowed.get(key) || allowed.get('runs')).has(name);
}

function drawBoard() {
  if (!board.runs) return;
  // the reference scored against itself is its own split, never averaged into the others
  const runs = board.runs.filter((r) => (board.split === 'gt' ? r.agent === 'gt'
    : r.agent !== 'gt' && (board.split === 'all' || r.kind === board.split)));
  const groups = new Map();      // "model × effort" -> case -> runs
  for (const r of runs) {
    const name = r.effort === null ? r.model : `${r.model} × ${r.effort}`;
    if (!groups.has(name)) groups.set(name, new Map());
    const cases = groups.get(name);
    const key = `${r.kind}/${r.case}`;
    if (!cases.has(key)) cases.set(key, []);
    cases.get(key).push(r);
  }
  // summarise every group of this split; the picker selects which are shown
  const summarised = [...groups].sort().map(([name, cases]) =>
    [name, [...cases].sort().map(([key, list]) => [key, caseSummary(list)])]);
  const chosen = (name) => board.only === null || board.only.has(name);
  const shown = summarised.filter(([name]) => chosen(name));
  drawBoardModels(summarised);
  const allow = board.basis === 'shared' && shown.length ? sharedCases(shown) : null;

  const held = (key) => shown.some(([, perCase]) =>
    perCase.some(([, c]) => Number.isFinite(c[key])));
  const count = (row, key) => {
    const n = row.n && row.n[basisKey(key)];
    return board.counts && n !== undefined ? ` <span class="n">${n}</span>` : '';
  };
  const cell = (row, key, digits) => {
    if (key === 'runs') return `${row.runs}`;
    if (key === 'executability') return `${row.executable}/${row.runs}`;
    const v = row[key];
    if (v === null || v === undefined) return '—';
    // the residual is shown as start → end
    if (key === 'residual' && row.residual_start !== null && row.residual_start !== undefined) {
      return `${row.residual_start.toFixed(digits)} → ${v.toFixed(digits)}${count(row, key)}`;
    }
    return `${v.toFixed(digits)}${count(row, key)}`;
  };
  // rows shared by all tables: one per group, plus one per case in By case mode
  const rows = [];
  for (const [name, perCase] of shown) {
    rows.push(['group', name, groupSummary(perCase, allow)]);
    if (board.rows !== 'case') continue;
    for (const [key, c] of perCase) {
      if (!allow || allow('runs', key)) rows.push(['case', key, c]);
    }
  }
  drawBoardNote(shown, allow);
  // a table holds at most BOARD_WIDTH readings; a longer group spans several tables
  const lines = [];
  BOARD_TABLES.forEach(([title, columns], index) => {
    const listed = index === 2 ? columns : columns.filter(([, key]) => held(key));
    if (!listed.length) return;
    const head = index === 0 ? [['runs', 'runs', 0], ['executable ↑', 'executability', 0]] : [];
    for (let start = 0; start < listed.length; start += BOARD_WIDTH) {
      const all = [...(start === 0 ? head : []), ...listed.slice(start, start + BOARD_WIDTH)];
      lines.push(`<h2>${title}</h2><table>`);
      lines.push(`<tr><th>model × effort</th>${all.map(([l]) => `<th>${l}</th>`).join('')}</tr>`);
      for (const [kind, name, row] of rows) {
        lines.push(`<tr class="${kind}"><td>${name}</td>${all.map(([, k, d]) => `<td>${cell(row, k, d)}</td>`).join('')}</tr>`);
      }
      lines.push('</table>');
    }
  });
  document.getElementById('board-tables').innerHTML = lines.join('')
    || '<p>no runs in this split</p>';
}

// The group picker: every group in this split with its case count, toggled on or off.
function drawBoardModels(summarised) {
  const host = document.getElementById('board-models');
  host.innerHTML = '';
  const chosen = (name) => board.only === null || board.only.has(name);
  const set = (names) => { board.only = new Set(names); drawBoard(); };
  for (const [name, perCase] of summarised) {
    const button = document.createElement('button');
    button.innerHTML = `${name} <span class="n">${perCase.length}</span>`;
    button.className = chosen(name) ? 'on' : '';
    button.onclick = () => {
      const kept = summarised.map(([other]) => other).filter(
        (other) => (other === name ? !chosen(other) : chosen(other)));
      set(kept.length ? kept : summarised.map(([other]) => other));
    };
    host.appendChild(button);
  }
  const all = document.createElement('button');
  all.textContent = 'All models';
  all.className = board.only === null ? 'on' : '';
  all.onclick = () => { board.only = null; drawBoard(); };
  host.appendChild(all);
}

// The note under the tables: the case coverage of the shown groups, flagged when it
// differs between them.
function drawBoardNote(shown, allow) {
  const note = document.getElementById('board-note');
  if (!shown.length) { note.textContent = ''; return; }
  const covered = shown.map(([name, perCase]) => [name, perCase.length]);
  const union = new Set(shown.flatMap(([, perCase]) => perCase.map(([key]) => key)));
  if (allow) {
    const kept = [...union].filter((key) => allow('runs', key)).length;
    note.className = '';
    note.textContent = `${kept} of ${union.size} cases are held by all ${shown.length} `
      + `shown groups; every average is taken over those alone, and a reading missing `
      + `from one group is dropped from all of them. The count after a number is the `
      + `cases it was averaged over.`;
    return;
  }
  const spread = new Set(covered.map(([, n]) => n));
  if (spread.size === 1) {
    note.className = '';
    note.textContent = `All ${shown.length} groups cover the same ${covered[0][1]} cases. `
      + `The count after a number is the cases it was averaged over.`;
    return;
  }
  note.className = 'bad';
  note.textContent = 'These groups cover different case sets — '
    + covered.map(([name, n]) => `${name}: ${n}`).join(', ')
    + ` of ${union.size} — so their averages are over different tasks and are not `
    + 'comparable. Switch Cases to Shared, or drop a group, to compare like with like.';
}

async function showBoard(open) {
  board.open = open;
  document.getElementById('board').style.display = open ? 'block' : 'none';
  document.getElementById('board-open').classList.toggle('on', open);
  if (!open) return;
  if (!board.runs) {
    try {
      board.runs = await (await fetch('/runs')).json();
    } catch (error) {
      document.getElementById('board-tables').textContent = `${error}`;
      return;
    }
  }
  drawBoard();
}

function boardControls() {
  document.getElementById('board-open').onclick = () => showBoard(!board.open);
  document.getElementById('board-close').onclick = () => showBoard(false);
  const splits = ['all', 'synthetic', 'real', 'gt'];
  buttons('board-split', ['All', 'Synthetic', 'Real', 'GT'], () => splits.indexOf(board.split),
    // a new split holds different groups, so the picker resets
    (i) => { board.split = splits[i]; board.only = null; drawBoard(); });
  const rows = ['average', 'case'];
  buttons('board-rows', ['Average', 'By case'], () => rows.indexOf(board.rows),
    (i) => { board.rows = rows[i]; drawBoard(); });
  const bases = ['all', 'shared'];
  buttons('board-basis', ['All', 'Shared'], () => bases.indexOf(board.basis),
    (i) => { board.basis = bases[i]; drawBoard(); });
  buttons('board-counts', ['Counts'], () => (board.counts ? 0 : -1),
    () => { board.counts = !board.counts; drawBoard(); });
}

let allWorlds = [];

async function start() {
  allWorlds = await (await fetch('/worlds')).json();

  const query = new URLSearchParams(location.search);
  // `view` names the panels, one or several: `?view=trajectory,meshes`
  if (query.has('view')) state.panels = query.get('view').split(',').filter(Boolean);
  if (query.has('space')) state.space = query.get('space');
  state.spaceChosen = query.has('space');
  boardControls();
  if (query.get('table') === '1') showBoard(true);
  buttons('colours', COLOUR_NAMES, () => COLOURS.indexOf(state.colour),
    (i) => { state.colour = COLOURS[i]; });
  drawSpaces();
  slider('trail', state.trail, (v) => { state.trail = v; for (const p of panels) applyFrame(p); },
         (v) => `${v} frames`);
  slider('rate', state.fps, (v) => { state.fps = v; }, (v) => `${v} fps`);
  // density changes rebuild every path, so they are applied once per animation frame
  slider('density', state.density, (v) => { state.density = v; queued.refresh = true; },
         (v) => `${v}% of points`);

  for (const name of ['gt', 'pred']) {
    const box = document.getElementById(`show-${name}`);
    box.onchange = () => { state.show[name] = box.checked; state.drawn = null; refresh(); };
  }
  const overlay = document.getElementById('overlay');
  overlay.checked = state.overlay;
  overlay.onchange = () => { state.overlay = overlay.checked; state.drawn = null; refresh(); };

  const scrub = document.getElementById('scrub');
  // scrub input is applied once per animation frame
  scrub.oninput = () => { state.frame = +scrub.value; queued.frame = true; };
  const play = document.getElementById('play');
  play.onclick = () => {
    state.playing = !state.playing;
    play.classList.toggle('on', state.playing);
    play.textContent = state.playing ? 'Pause' : 'Play';
  };
  document.getElementById('take-toggle').onclick = () => {
    take.open = !take.open;
    syncTake();
    remedia();
    plot.marked = plot.said = null;      // folded away, the marker stopped moving
    markPlot();
  };
  document.getElementById('plot-pick').onchange = (e) => {
    plot.key = e.target.value;
    drawPlot();
  };
  document.getElementById('scene').onchange = (e) => {
    state.case = e.target.value;
    const first = allWorlds.find((row) => row.case === state.case);
    if (first) load(first.world);
  };
  document.getElementById('world').onchange = (e) => load(e.target.value);

  resize();
  const wanted = allWorlds.find((row) => row.world === query.get('world')) || allWorlds[0];
  if (wanted) {
    state.case = wanted.case;
    state.world = wanted.world;
    drawPickers();
    await load(wanted.world);
    if (query.has('frame')) {
      state.frame = +query.get('frame');
      scrub.value = state.frame;
      await refresh();
    }
  }
  tick();
}

const SIZE = new THREE.Vector2();
let last = 0;
// work requested by controls, applied on the next animation frame
const queued = { frame: false, refresh: false };

// Render every 3D panel into the rectangle of its DOM body, measured each frame, and clear
// the gaps between panels to the page colour.
function renderPanels() {
  // measure every rectangle before writing any, to avoid a forced reflow per panel
  const host = stage.getBoundingClientRect();
  const boxes = panels.map(
    (panel) => (panel.scene ? panel.body.getBoundingClientRect() : null));
  renderer.setScissorTest(false);
  renderer.setClearColor(GAP_INK, 1);
  renderer.clear();
  renderer.setScissorTest(true);
  for (const [index, panel] of panels.entries()) {
    const box = boxes[index];
    if (!box || box.width < 4 || box.height < 4) continue;
    const left = box.left - host.left, top = box.top - host.top;
    // the viewport is the whole panel; the scissor is the part inside the stage, so a
    // partly scrolled-out panel is drawn partly
    const clipped = { x0: Math.max(0, left), y0: Math.max(0, top),
                      x1: Math.min(host.width, left + box.width),
                      y1: Math.min(host.height, top + box.height) };
    if (clipped.x1 - clipped.x0 < 1 || clipped.y1 - clipped.y0 < 1) continue;
    const aspect = box.width / box.height;
    if (panel.camera.aspect !== aspect) {
      panel.camera.aspect = aspect;
      panel.camera.updateProjectionMatrix();
    }
    panel.controls.update();
    clip(panel);
    ruler(panel, box.height);
    renderer.setViewport(left, host.height - (top + box.height), box.width, box.height);
    renderer.setScissor(clipped.x0, host.height - clipped.y1,
                        clipped.x1 - clipped.x0, clipped.y1 - clipped.y0);
    // report a render error, such as a shader link failure, and stop playback
    try {
      renderer.render(panel.scene, panel.camera);
    } catch (error) {
      state.playing = false;
      note(`WebGL: ${error.message}`);
      console.error(error);
      return;
    }
  }
}

// Whether a video leads playback, which needs a known duration. Otherwise the frame counter
// advances on its own, so a grid of 3D panels still plays.
function leading() {
  return media.length > 0 && Number.isFinite(media[0].duration) && media[0].duration > 0;
}

function tick(now = 0) {
  requestAnimationFrame(tick);
  const scrub = document.getElementById('scrub');
  if (queued.refresh) { queued.refresh = false; refresh(); }
  if (queued.frame) { queued.frame = false; showFrame(); }
  if (leading()) {
    // the leading video is the clock; the rest of the grid follows its frame
    const before = state.frame;
    playMedia(scrub);
    if (state.frame !== before) {
      for (const panel of panels) applyFrame(panel);
      frameText();
      markPlot();
      drawGeometry(state.frame);
    }
  } else if (state.playing && now - last > 1000 / state.fps) {
    last = now;
    state.frame = (state.frame + 1) % (+scrub.max + 1);
    scrub.value = state.frame;
    showFrame();
  }
  if (!state.playing) for (const video of media) if (!video.paused) video.pause();
  const size = renderer.getSize(SIZE);
  if (Math.abs(size.x - stage.clientWidth) > 1
      || Math.abs(size.y - stage.clientHeight) > 1) resize();
  renderPanels();
}

start();

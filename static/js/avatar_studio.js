/*
 * AvatarStudio — 3D try-on avatar for StyleMate (three.js r128: global THREE, GLTFLoader, OrbitControls).
 *
 *  Outfit model:  the avatar wears a set of slots (top, outer, bottom, dress, shoes, accessories).
 *                 applyItem() updates the slots with dressing rules (a dress replaces top + bottom,
 *                 a top or bottom replaces a dress, ...) and render() rebuilds every visual from the
 *                 slots, so any order or combination of items ends in a consistent look.
 *  Body:          one natural standing stance (A-pose rig relaxed to arms-down) with breathing and
 *                 blinking; bones are scaled from the measured proportions (incl. child proportions).
 *  Garments:      the outfit meshes are recoloured with fabric-specific finish; t-shirts / shorts /
 *                 capris are cut to show skin; jackets and blazers are a separate open-front layer
 *                 over the shirt; fit ease inflates/shrinks the torso and shows a heat-map.
 *  Cloth:         skirts, dresses and kurta hems are a verlet cloth pinned at the waist, colliding with
 *                 pelvis, legs, forearms and hands, stepped at a fixed 60 Hz, reacting to wind.
 */
(function () {
  'use strict';

  // ---------------------------------------------------------------- fabric presets
  // stiff: stretch/bend resistance (0..1), gsm: weight, sway: hem flutter, damp: cloth velocity damping
  const FABRICS = {
    cotton:   { stiff: 0.55, bend: 0.35, gsm: 150, sway: 0.35, damp: 0.985, rough: 0.9,  metal: 0.0 },
    denim:    { stiff: 0.85, bend: 0.70, gsm: 350, sway: 0.08, damp: 0.97,  rough: 0.95, metal: 0.0 },
    linen:    { stiff: 0.65, bend: 0.45, gsm: 180, sway: 0.30, damp: 0.985, rough: 0.92, metal: 0.0 },
    silk:     { stiff: 0.20, bend: 0.08, gsm: 60,  sway: 1.00, damp: 0.993, rough: 0.35, metal: 0.08 },
    satin:    { stiff: 0.25, bend: 0.10, gsm: 90,  sway: 0.90, damp: 0.992, rough: 0.3,  metal: 0.1 },
    wool:     { stiff: 0.60, bend: 0.50, gsm: 280, sway: 0.15, damp: 0.98,  rough: 1.0,  metal: 0.0 },
    knit:     { stiff: 0.30, bend: 0.25, gsm: 200, sway: 0.30, damp: 0.985, rough: 1.0,  metal: 0.0 },
    leather:  { stiff: 0.95, bend: 0.85, gsm: 900, sway: 0.03, damp: 0.96,  rough: 0.45, metal: 0.05 },
    suede:    { stiff: 0.85, bend: 0.75, gsm: 700, sway: 0.04, damp: 0.96,  rough: 1.0,  metal: 0.0 },
    polyester:{ stiff: 0.45, bend: 0.25, gsm: 120, sway: 0.45, damp: 0.987, rough: 0.6,  metal: 0.02 },
    chiffon:  { stiff: 0.12, bend: 0.05, gsm: 40,  sway: 1.20, damp: 0.994, rough: 0.55, metal: 0.0 },
    velvet:   { stiff: 0.50, bend: 0.40, gsm: 300, sway: 0.15, damp: 0.98,  rough: 0.85, metal: 0.05 },
    fleece:   { stiff: 0.40, bend: 0.35, gsm: 260, sway: 0.15, damp: 0.98,  rough: 1.0,  metal: 0.0 },
    corduroy: { stiff: 0.70, bend: 0.55, gsm: 320, sway: 0.10, damp: 0.975, rough: 0.95, metal: 0.0 },
    lace:     { stiff: 0.15, bend: 0.06, gsm: 70,  sway: 0.90, damp: 0.993, rough: 0.8,  metal: 0.0 },
    nylon:    { stiff: 0.35, bend: 0.15, gsm: 80,  sway: 0.60, damp: 0.99,  rough: 0.5,  metal: 0.03 },
  };
  const fabricOf = (name) => FABRICS[name] || FABRICS.cotton;
  // r128 renders in sRGB but takes material colours as linear
  const lin = (hex) => new THREE.Color(hex).convertSRGBToLinear();
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  const SKIN_TONES = [
    { name: 'Natural', tint: '#ffffff', skin: '#d9a382' }, { name: 'Fair ivory', tint: '#ffe4d6', skin: '#f1c7ad' },
    { name: 'Olive', tint: '#e8be9e', skin: '#c99a72' }, { name: 'Bronze', tint: '#b37d5c', skin: '#9c6b4c' },
    { name: 'Deep', tint: '#6b3a2a', skin: '#5e3a28' },
  ];
  const DEFAULT_TOP = { slot: 'top', name: 'Base shirt', color: '#4f46e5', fabric: 'cotton', article_type: 'Shirts', base: true };
  const DEFAULT_BOTTOM = { slot: 'bottom', name: 'Base trousers', color: '#1f2937', fabric: 'denim', article_type: 'Trousers', base: true };
  const DEFAULT_SHOES = { slot: 'shoes', name: 'Base shoes', color: '#0f172a', fabric: 'leather', base: true };
  const ACCESSORY_SLOTS = ['sunglasses', 'watch', 'cap', 'belt', 'bag'];

  const S = {
    ready: false, container: null, scene: null, camera: null, renderer: null, controls: null, clock: null,
    avatar: null, bones: {}, rest: {}, meshes: {}, bind: {}, uniforms: {}, acc: {}, outer: null,
    outfit: {}, fit: null, time: 0, acc60: 0, armOut: 0.13, armOutTarget: 0.13,
    wind: 0, windDir: new THREE.Vector3(0.35, 0, -1).normalize(), skinIndex: 0, heat: false,
    body: { s: 1, hipK: 1 }, cloth: null, onStatus: () => {}, onOutfit: () => {}, blinkT: 2, smileT: 0,
  };

  const _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion(), _v = new THREE.Vector3(), _v2 = new THREE.Vector3(),
    _v3 = new THREE.Vector3();
  const AX = { x: new THREE.Vector3(1, 0, 0), y: new THREE.Vector3(0, 1, 0), z: new THREE.Vector3(0, 0, 1) };
  const AXN = ['x', 'y', 'z'];

  // ---------------------------------------------------------------- bone helpers
  function rotWorld(bone, axisWorld, angle) {
    if (!bone || !angle) return;
    bone.parent.getWorldQuaternion(_q);
    _q2.setFromAxisAngle(axisWorld, angle);
    bone.quaternion.premultiply(_q.clone().invert().multiply(_q2).multiply(_q));
  }
  function worldPos(name, target) { return S.bones[name].getWorldPosition(target || new THREE.Vector3()); }
  function localAxisFor(bone, worldAxis) {
    bone.getWorldQuaternion(_q);
    const a = worldAxis.clone().applyQuaternion(_q.clone().invert());
    const c = [Math.abs(a.x), Math.abs(a.y), Math.abs(a.z)];
    return c.indexOf(Math.max(...c));
  }

  // ---------------------------------------------------------------- garment shader
  function makeUniforms(kind) {
    return {
      uTime: { value: 0 }, uSway: { value: 0.3 }, uWind: { value: new THREE.Vector3() },
      uHem: { value: kind === 'bottom' ? new THREE.Vector2(0.12, 0.5) : new THREE.Vector2(1.03, 1.16) },
      uBandY: { value: new THREE.Vector3(1.35, 1.12, 0.95) }, uBandW: { value: 0.075 },
      uEase: { value: new THREE.Vector3() }, uRegOn: { value: new THREE.Vector3() },
      uHeatOn: { value: 0 }, uHeatA: { value: new THREE.Color() }, uHeatB: { value: new THREE.Color() }, uHeatC: { value: new THREE.Color() },
      uInflate: { value: 0 }, uShrink: { value: 0 }, uHideAbove: { value: 10 }, uHideBelow: { value: -10 },
      uCutY: { value: -10 }, uSleeveX: { value: 10 }, uOpenFront: { value: 0 }, uSkin: { value: lin(SKIN_TONES[0].skin) },
      uBeltOn: { value: 0 }, uBelt: { value: new THREE.Color() }, uBeltY: { value: new THREE.Vector2(1.033, 1.078) },
    };
  }

  function patchGarment(material, kind) {
    const U = makeUniforms(kind);
    S.uniforms[kind] = U;
    material.customProgramCacheKey = () => 'stylemate-garment-v2';
    material.onBeforeCompile = (shader) => {
      Object.assign(shader.uniforms, U);
      shader.vertexShader = shader.vertexShader
        .replace('#include <common>', `#include <common>
          uniform float uTime; uniform float uSway; uniform vec3 uWind; uniform vec2 uHem;
          uniform vec3 uBandY; uniform float uBandW; uniform vec3 uEase; uniform vec3 uRegOn;
          uniform float uInflate; uniform float uShrink; uniform float uCutY;
          varying vec3 vBind;`)
        .replace('#include <skinning_vertex>', `#include <skinning_vertex>
          vBind = position;
          vec3 n = normalize(objectNormal);
          // fit ease on the torso only (sleeves keep their size so arms never poke through)
          float torso = 1.0 - smoothstep(0.16, 0.21, abs(position.x));
          vec3 d = (vec3(position.y) - uBandY) / uBandW;
          vec3 w = exp(-d * d) * uRegOn;
          float ws = w.x + w.y + w.z;
          float ease = (dot(w, uEase) / max(ws, 1e-3)) * clamp(ws, 0.0, 1.0) * torso;
          // bare legs below a shorts hem are slimmer than the trouser leg; keep the hem over the shoe collar
          float bare = 1.0 - smoothstep(uCutY - 0.03, uCutY, position.y);
          float aboveShoe = smoothstep(0.13, 0.24, position.y);
          float offset = uInflate + ease + (uShrink - 0.02 * bare) * aboveShoe;
          // hem / cuff flutter: only ever outward or sideways, never into the body
          float hem = 1.0 - smoothstep(uHem.x, uHem.y, position.y);
          float cuff = smoothstep(0.32, 0.42, abs(position.x));
          float m = max(hem, cuff * 0.6) * uSway;
          float ph = uTime * 2.4 + position.y * 23.0 + position.x * 11.0;
          vec3 flutter = (uWind * (0.65 + 0.35 * sin(uTime * 0.8 + position.y * 3.0))
                          + vec3(sin(ph), 0.0, cos(ph * 0.9)) * (0.12 + length(uWind)) * 0.3) * 0.016 * m;
          float inward = min(dot(flutter, n), 0.0);
          flutter -= n * inward;
          transformed += n * offset + flutter;`);
      shader.fragmentShader = shader.fragmentShader
        .replace('#include <common>', `#include <common>
          uniform float uHeatOn; uniform vec3 uHeatA; uniform vec3 uHeatB; uniform vec3 uHeatC;
          uniform vec3 uBandY; uniform float uBandW; uniform vec3 uRegOn;
          uniform float uCutY; uniform float uSleeveX; uniform vec3 uSkin; uniform float uHideAbove; uniform float uHideBelow;
          uniform float uOpenFront; uniform float uBeltOn; uniform vec3 uBelt; uniform vec2 uBeltY;
          varying vec3 vBind;`)
        .replace('#include <color_fragment>', `#include <color_fragment>
          if (vBind.y > uHideAbove || vBind.y < uHideBelow) discard;          // tucked / covered parts
          // open jacket: a front opening that widens into lapels toward the collar
          if (uOpenFront > 0.5 && vBind.z > 0.0 && vBind.y < 1.5 &&
              abs(vBind.x) < 0.03 + 0.2 * max(vBind.y - 1.2, 0.0)) discard;
          if (uBeltOn > 0.5 && vBind.y > uBeltY.x && vBind.y < uBeltY.y) diffuseColor.rgb = uBelt;  // modelled belt
          else if (vBind.y < uCutY || abs(vBind.x) > uSleeveX) diffuseColor.rgb = uSkin;            // shorts / short sleeves
          else if (uHeatOn > 0.5) {
            vec3 d = (vec3(vBind.y) - uBandY) / (uBandW * 1.25);
            vec3 w = exp(-d * d) * uRegOn * (1.0 - smoothstep(0.16, 0.21, abs(vBind.x)));
            float ws = w.x + w.y + w.z;
            vec3 hc = (w.x * uHeatA + w.y * uHeatB + w.z * uHeatC) / max(ws, 1e-3);
            diffuseColor.rgb = mix(diffuseColor.rgb * 0.55 + 0.25, hc, clamp(ws, 0.0, 1.0) * 0.85);
          }`);
    };
    material.needsUpdate = true;
    return U;
  }

  // ---------------------------------------------------------------- verlet cloth (skirts, dresses, kurta hems)
  const CLOTH_THICKNESS = 0.012;   // metres kept between cloth and any collider

  class Cloth {
    constructor(spec) {
      this.spec = spec;
      this.cols = 32; this.rows = 16;
      const f = fabricOf(spec.fabric);
      this.f = f;
      this.iter = 4 + Math.round(f.stiff * 8);
      const n = this.cols * (this.rows + 1);
      this.p = new Float32Array(n * 3); this.o = new Float32Array(n * 3);
      const hips = S.bones.Hips;
      hips.updateWorldMatrix(true, false);
      const inv = new THREE.Matrix4().copy(hips.matrixWorld).invert();
      const s = S.body.s, k = S.body.hipK;
      const centre = worldPos('Hips');
      const rx = S.bind.hipHalf * 1.3 * k * s, rz = S.bind.hipDepth * 1.35 * k * s;
      this.ring = [];       // pinned waistband, stored in Hips-local space so it follows the body
      for (let c = 0; c < this.cols; c++) {
        const a = Math.PI + (c / this.cols) * Math.PI * 2;     // seam at the back
        this.ring.push(new THREE.Vector3(centre.x + Math.sin(a) * rx, centre.y + S.bind.waistAboveHips * s,
          centre.z + Math.cos(a) * rz + 0.01 * s).applyMatrix4(inv));
      }
      this.len = spec.length * s;
      const flare = spec.flare * (0.3 + 0.7 * f.stiff);       // stiff fabrics stand out, silk falls close
      for (let r = 0; r <= this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          const i = (r * this.cols + c) * 3, t = r / this.rows;
          const top = this.ring[c].clone().applyMatrix4(hips.matrixWorld);
          const grow = 1 + flare * t;
          this.p[i] = centre.x + (top.x - centre.x) * grow;
          this.p[i + 1] = top.y - t * this.len;
          this.p[i + 2] = centre.z + (top.z - centre.z) * grow;
        }
      }
      this.o.set(this.p);
      this.cons = [];
      const idx = (r, c) => r * this.cols + ((c + this.cols) % this.cols);
      const dist = (i, j) => Math.hypot(this.p[i * 3] - this.p[j * 3], this.p[i * 3 + 1] - this.p[j * 3 + 1], this.p[i * 3 + 2] - this.p[j * 3 + 2]);
      for (let r = 0; r <= this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          const i = idx(r, c);
          this.cons.push([i, idx(r, c + 1), dist(i, idx(r, c + 1)), 1]);
          this.cons.push([i, idx(r, c + 2), dist(i, idx(r, c + 2)), f.bend * 0.6]);
          if (r < this.rows) {
            this.cons.push([i, idx(r + 1, c), dist(i, idx(r + 1, c)), 1]);
            this.cons.push([i, idx(r + 1, c + 1), dist(i, idx(r + 1, c + 1)), 0.5]);
          }
          if (r < this.rows - 1) this.cons.push([i, idx(r + 2, c), dist(i, idx(r + 2, c)), f.bend]);
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(this.p, 3));
      const ind = [];
      for (let r = 0; r < this.rows; r++) for (let c = 0; c < this.cols; c++) {
        const a = idx(r, c), b = idx(r, c + 1), d = idx(r + 1, c), e = idx(r + 1, c + 1);
        ind.push(a, d, b, b, d, e);
      }
      geo.setIndex(ind);
      this.mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
        color: lin(spec.color || '#888888'), roughness: f.rough, metalness: f.metal, side: THREE.DoubleSide }));
      this.mesh.castShadow = true; this.mesh.receiveShadow = true; this.mesh.frustumCulled = false;
      S.scene.add(this.mesh);
      for (let i = 0; i < 150; i++) this.step(1 / 60);        // drape before the first frame
      this.upload();
    }

    colliders() {
      const s = S.body.s, k = S.body.hipK, T = CLOTH_THICKNESS * s;
      const cap = (a, b, r) => ({ a: worldPos(a), b: worldPos(b), r: r * s + T });
      return {
        caps: [
          // legs: radii match the trouser mesh (wider than the bare leg)
          cap('LeftUpLeg', 'LeftLeg', 0.1 * k), cap('RightUpLeg', 'RightLeg', 0.1 * k),
          cap('LeftLeg', 'LeftFoot', 0.07), cap('RightLeg', 'RightFoot', 0.07),
          { a: worldPos('Hips'), b: worldPos('LeftLeg').add(worldPos('RightLeg')).multiplyScalar(0.5), r: 0.075 * k * s + T },
          // forearms and hands, so hands resting at the sides never sink into the skirt
          cap('LeftForeArm', 'LeftHand', 0.045), cap('RightForeArm', 'RightHand', 0.045),
          cap('LeftHand', 'LeftHandMiddle2', 0.045), cap('RightHand', 'RightHandMiddle2', 0.045),
        ],
        hip: { c: worldPos('Hips'), r: new THREE.Vector3(S.bind.hipHalf * 1.25 * k * s + T, 0.17 * s, S.bind.hipDepth * 1.2 * k * s + T) },
      };
    }

    step(dt) {
      const hips = S.bones.Hips; hips.updateWorldMatrix(true, false);
      const n = this.cols * (this.rows + 1), f = this.f, g = -9.8 * dt * dt;
      const windAcc = (S.wind * 6.0) / Math.max(0.35, f.gsm / 150);
      for (let i = 0; i < n; i++) {
        const j = i * 3;
        if (i < this.cols) {          // waistband pinned to the hips
          _v.copy(this.ring[i]).applyMatrix4(hips.matrixWorld);
          this.p[j] = this.o[j] = _v.x; this.p[j + 1] = this.o[j + 1] = _v.y; this.p[j + 2] = this.o[j + 2] = _v.z;
          continue;
        }
        const x = this.p[j], y = this.p[j + 1], z = this.p[j + 2];
        const gust = windAcc * (0.6 + 0.4 * Math.sin(S.time * 1.3 + y * 4.0 + x * 3.0)) * dt * dt;
        this.p[j] += (x - this.o[j]) * f.damp + S.windDir.x * gust;
        this.p[j + 1] += (y - this.o[j + 1]) * f.damp + g;
        this.p[j + 2] += (z - this.o[j + 2]) * f.damp + S.windDir.z * gust;
        this.o[j] = x; this.o[j + 1] = y; this.o[j + 2] = z;
      }
      const col = this.colliders();
      for (let it = 0; it < this.iter; it++) {
        for (const [a, b, rest, k] of this.cons) {
          const ia = a * 3, ib = b * 3;
          const dx = this.p[ib] - this.p[ia], dy = this.p[ib + 1] - this.p[ia + 1], dz = this.p[ib + 2] - this.p[ia + 2];
          const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6;
          const diff = ((d - rest) / d) * k * (rest > d ? 0.25 : 1);   // resists stretch more than compression
          const wa = a < this.cols ? 0 : 1, wb = b < this.cols ? 0 : 1, wsum = wa + wb;
          if (!wsum) continue;
          this.p[ia] += dx * diff * wa / wsum; this.p[ia + 1] += dy * diff * wa / wsum; this.p[ia + 2] += dz * diff * wa / wsum;
          this.p[ib] -= dx * diff * wb / wsum; this.p[ib + 1] -= dy * diff * wb / wsum; this.p[ib + 2] -= dz * diff * wb / wsum;
        }
        this.collide(col);
      }
    }

    collide(col) {
      const n = this.cols * (this.rows + 1);
      const h = col.hip;
      for (let i = this.cols; i < n; i++) {
        const j = i * 3;
        _v.set(this.p[j], this.p[j + 1], this.p[j + 2]);
        const qx = (_v.x - h.c.x) / h.r.x, qy = (_v.y - h.c.y) / h.r.y, qz = (_v.z - h.c.z) / h.r.z;
        const qd = Math.sqrt(qx * qx + qy * qy + qz * qz);
        if (qd < 1 && qd > 1e-6) _v.set(h.c.x + qx / qd * h.r.x, h.c.y + qy / qd * h.r.y, h.c.z + qz / qd * h.r.z);
        for (let pass = 0; pass < 3; pass++) {               // repeat: pushing out of one limb can enter another
          for (const c of col.caps) {
            _v2.subVectors(c.b, c.a);
            const t = clamp(_v3.subVectors(_v, c.a).dot(_v2) / _v2.lengthSq(), 0, 1);
            _v3.copy(c.a).addScaledVector(_v2, t);            // closest point on the bone segment
            const d = _v.distanceTo(_v3);
            if (d < c.r) _v.sub(_v3).multiplyScalar(c.r / (d || 1e-6)).add(_v3);
          }
        }
        if (_v.y < 0.01) _v.y = 0.01;
        this.p[j] = _v.x; this.p[j + 1] = _v.y; this.p[j + 2] = _v.z;
      }
    }

    upload() { this.mesh.geometry.attributes.position.needsUpdate = true; this.mesh.geometry.computeVertexNormals(); }
    dispose() { S.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh.material.dispose(); }
  }

  // ---------------------------------------------------------------- setup
  function init(container, opts = {}) {
    S.container = container;
    S.onStatus = opts.onStatus || (() => {});
    S.onOutfit = opts.onOutfit || (() => {});
    const badge = document.createElement('div');
    Object.assign(badge.style, { position: 'absolute', top: '14px', left: '14px', background: 'rgba(15,23,42,0.85)', color: '#fff',
      padding: '7px 13px', borderRadius: '20px', fontSize: '12px', fontWeight: '600', zIndex: '10', pointerEvents: 'none' });
    badge.textContent = '⏳ Loading 3D model…';
    container.appendChild(badge);

    S.scene = new THREE.Scene();
    S.camera = new THREE.PerspectiveCamera(38, container.clientWidth / container.clientHeight, 0.05, 100);
    S.camera.position.set(0.6, 1.25, 2.6);
    S.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: !!opts.preserveDrawingBuffer });
    S.renderer.setSize(container.clientWidth, container.clientHeight);
    S.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    S.renderer.shadowMap.enabled = true;
    S.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    S.renderer.outputEncoding = THREE.sRGBEncoding;
    // linear tone mapping keeps garment colours close to the catalog colour (ACES over-saturated dark fabrics)
    S.renderer.toneMapping = THREE.LinearToneMapping;
    S.renderer.toneMappingExposure = 1.0;
    container.appendChild(S.renderer.domElement);

    S.controls = new THREE.OrbitControls(S.camera, S.renderer.domElement);
    S.controls.enableDamping = true; S.controls.dampingFactor = 0.06;
    S.controls.target.set(0, 0.95, 0); S.controls.minDistance = 0.5; S.controls.maxDistance = 5;
    S.controls.maxPolarAngle = Math.PI / 2 + 0.05;

    S.scene.add(new THREE.HemisphereLight(0xfff7ed, 0x334155, 0.7));
    const key = new THREE.DirectionalLight(0xfff5ea, 1.0);
    key.position.set(2.4, 3.6, 3.2); key.castShadow = true; key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -1.5; key.shadow.camera.right = 1.5; key.shadow.camera.top = 2.5; key.shadow.camera.bottom = -0.5;
    S.scene.add(key);
    const fill = new THREE.DirectionalLight(0xe0e7ff, 0.45); fill.position.set(-2.5, 2.2, 1.5); S.scene.add(fill);
    const rim = new THREE.DirectionalLight(0x93c5fd, 0.5); rim.position.set(0, 2.8, -2.8); S.scene.add(rim);
    const podium = new THREE.Mesh(new THREE.CylinderGeometry(0.85, 0.95, 0.06, 64),
      new THREE.MeshStandardMaterial({ color: 0x161f30, roughness: 0.25, metalness: 0.6 }));
    podium.position.y = -0.03; podium.receiveShadow = true; S.scene.add(podium);
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.88, 0.91, 64), new THREE.MeshBasicMaterial({ color: 0x6366f1, side: THREE.DoubleSide }));
    ring.rotation.x = Math.PI / 2; ring.position.y = 0.002; S.scene.add(ring);

    new THREE.GLTFLoader().load(opts.modelUrl || '/web_app/models/realistic_human.glb', (gltf) => {
      onModel(gltf.scene);
      badge.textContent = '✓ 3D model ready';
      setTimeout(() => { badge.style.transition = 'opacity .6s'; badge.style.opacity = '0'; }, 1200);
      if (opts.onReady) opts.onReady();
    }, (xhr) => {
      if (xhr.lengthComputable) badge.textContent = `⏳ Loading 3D model (${Math.round(xhr.loaded / xhr.total * 100)}%)…`;
    }, (err) => {
      console.error('[AvatarStudio] load error', err);
      badge.textContent = '⚠️ Could not load the 3D model';
      badge.style.background = 'rgba(239,68,68,0.9)';
    });

    window.addEventListener('resize', resize);
    S.clock = new THREE.Clock();
    S.renderer.setAnimationLoop(frame);
  }

  function resize() {
    const c = S.container;
    S.camera.aspect = c.clientWidth / c.clientHeight;
    S.camera.updateProjectionMatrix();
    S.renderer.setSize(c.clientWidth, c.clientHeight);
  }

  function onModel(root) {
    S.avatar = root;
    root.traverse((o) => {
      if (o.isBone) S.bones[o.name] = o;
      if (o.isMesh) {
        o.castShadow = o.receiveShadow = true;
        o.frustumCulled = false;            // skinned + displaced meshes: bounding boxes are wrong
        if (o.material) { o.material = o.material.clone(); o.userData.normalMap = o.material.normalMap || null; }
        S.meshes[o.name] = o;
      }
    });
    S.scene.add(root);
    root.updateMatrixWorld(true);
    for (const [name, b] of Object.entries(S.bones)) S.rest[name] = { q: b.quaternion.clone(), p: b.position.clone(), s: b.scale.clone() };
    const P = (n) => worldPos(n);
    S.bind.height = P('HeadTop_End').y;
    S.bind.shoulder = P('LeftArm').distanceTo(P('RightArm'));
    S.bind.hipHalf = P('LeftUpLeg').distanceTo(P('RightUpLeg')) / 2 + 0.035;
    S.bind.hipDepth = 0.11;
    S.bind.hipsY = P('Hips').y;
    S.bind.waistAboveHips = P('Spine').y - P('Hips').y + 0.03;
    S.bind.waistY = S.bind.hipsY + S.bind.waistAboveHips;          // where skirts / dress skirts are pinned
    S.bind.leg = P('LeftUpLeg').y - P('LeftFoot').y;
    S.bind.arm = P('LeftArm').distanceTo(P('LeftForeArm')) + P('LeftForeArm').distanceTo(P('LeftHand'));
    S.bind.toeY = Math.min(P('LeftToe_End').y, P('RightToe_End').y);
    const dl = P('LeftForeArm').sub(P('LeftArm')), dr = P('RightForeArm').sub(P('RightArm'));
    S.bind.armAngL = Math.atan2(dl.x, -dl.y);      // 0 = hanging straight down, + = outward
    S.bind.armAngR = Math.atan2(-dr.x, -dr.y);

    const top = S.meshes.Wolf3D_Outfit_Top;
    patchGarment(top.material, 'top');
    patchGarment(S.meshes.Wolf3D_Outfit_Bottom.material, 'bottom');
    // jackets/blazers: a second, slightly larger copy of the top mesh on the same skeleton
    const outer = new THREE.SkinnedMesh(top.geometry, top.material.clone());
    outer.name = 'StyleMate_Outer';
    outer.castShadow = outer.receiveShadow = true;
    outer.frustumCulled = false;
    outer.userData.normalMap = top.userData.normalMap;
    outer.position.copy(top.position); outer.quaternion.copy(top.quaternion); outer.scale.copy(top.scale);
    patchGarment(outer.material, 'outer');
    S.uniforms.outer.uInflate.value = 0.016;
    S.uniforms.outer.uOpenFront.value = 1;
    outer.visible = false;
    top.parent.add(outer);
    outer.bind(top.skeleton, top.bindMatrix);     // same bones as the shirt, so it moves with it
    S.outer = outer;
    for (const U of Object.values(S.uniforms)) U.uBandY.value.set(P('Spine2').y + 0.06, P('Spine').y + 0.04, P('Hips').y - 0.04);

    buildAccessories();
    S.ready = true;
    resetOutfit(true);
  }

  // ---------------------------------------------------------------- accessories (procedural, bone-attached)
  function buildAccessories() {
    const mat = (c, r = 0.5, m = 0) => new THREE.MeshStandardMaterial({ color: lin(c), roughness: r, metalness: m });
    const part = (geo, m, x, y, z) => { const o = new THREE.Mesh(geo, m); o.position.set(x, y, z); o.castShadow = true; return o; };
    const attach = (boneName, group) => { S.bones[boneName].attach(group); group.visible = false; return group; };

    // sunglasses at the eyes, on the head bone
    const g = new THREE.Group(); const frameM = mat('#d4a017', 0.2, 0.9), lensM = mat('#0b1220', 0.1, 0.9);
    [-1, 1].forEach((s) => {
      g.add(part(new THREE.BoxGeometry(0.048, 0.028, 0.01), lensM, s * 0.033, 0, 0));
      g.add(part(new THREE.BoxGeometry(0.052, 0.032, 0.008), frameM, s * 0.033, 0, -0.002));
      g.add(part(new THREE.BoxGeometry(0.004, 0.004, 0.11), frameM, s * 0.074, 0.008, -0.05));
    });
    g.add(part(new THREE.BoxGeometry(0.02, 0.005, 0.008), frameM, 0, 0.006, 0));
    g.position.copy(worldPos('LeftEye').add(worldPos('RightEye')).multiplyScalar(0.5)).add(new THREE.Vector3(0, 0, 0.035));
    S.acc.sunglasses = { group: attach('Head', g), tint: [frameM] };

    // cap: crown + brim above the head
    const c = new THREE.Group(); const capM = mat('#1e3a8a', 0.9);
    // crown is a half-ellipsoid that covers the hair down to the forehead; the brim points forward
    const crown = part(new THREE.SphereGeometry(0.1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), capM, 0, 0, 0);
    crown.scale.set(1.0, 0.95, 1.12);
    const brim = part(new THREE.CylinderGeometry(0.085, 0.085, 0.007, 24, 1, false, -Math.PI / 2, Math.PI), capM, 0, 0.004, 0.085);
    brim.scale.set(1, 1, 0.9);
    brim.rotation.x = 0.12;
    const band = part(new THREE.CylinderGeometry(0.101, 0.101, 0.018, 32, 1, true), capM, 0, 0.008, 0);
    band.scale.set(1.0, 1, 1.12);
    c.add(crown, brim, band);
    c.position.copy(worldPos('HeadTop_End')).add(new THREE.Vector3(0, -0.105, -0.005));
    c.rotation.x = -0.08;
    S.acc.cap = { group: attach('Head', c), tint: [capM] };

    // watch on the left wrist
    const w = new THREE.Group(); const strapM = mat('#1e293b', 0.6), dialM = mat('#d4a017', 0.2, 0.9);
    w.add(new THREE.Mesh(new THREE.CylinderGeometry(0.031, 0.031, 0.022, 24, 1, true), strapM));
    strapM.side = THREE.DoubleSide;
    const dial = part(new THREE.CylinderGeometry(0.017, 0.017, 0.008, 24), dialM, 0, 0, 0.03);
    dial.rotation.x = Math.PI / 2; w.add(dial);
    w.position.copy(worldPos('LeftHand').lerp(worldPos('LeftForeArm'), 0.12));
    w.quaternion.setFromUnitVectors(AX.y, worldPos('LeftHand').sub(worldPos('LeftForeArm')).normalize());
    S.acc.watch = { group: attach('LeftHand', w), tint: [strapM] };

    // handbag hanging from the right hand
    const h = new THREE.Group(); const bagM = mat('#7c4a2d', 0.55);
    h.add(part(new THREE.BoxGeometry(0.2, 0.16, 0.07), bagM, 0, -0.13, 0));
    const handle = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.008, 8, 24, Math.PI), bagM);
    handle.position.set(0, -0.05, 0); h.add(handle);
    h.position.copy(worldPos('RightHand')).add(new THREE.Vector3(-0.02, -0.02, 0.0));
    S.acc.bag = { group: attach('RightHand', h), tint: [bagM] };

    // backpack on the upper back
    const k = new THREE.Group(); const packM = mat('#111827', 0.7);
    k.add(part(new THREE.BoxGeometry(0.3, 0.4, 0.13), packM, 0, 0, 0));
    k.add(part(new THREE.BoxGeometry(0.22, 0.12, 0.05), packM, 0, -0.1, -0.08));      // front pocket
    [-1, 1].forEach((s) => {
      k.add(part(new THREE.BoxGeometry(0.045, 0.012, 0.2), packM, s * 0.1, 0.235, 0.135));   // over the shoulder
      k.add(part(new THREE.BoxGeometry(0.045, 0.24, 0.012), packM, s * 0.1, 0.12, 0.24));     // down the chest
    });
    k.position.copy(worldPos('Spine2')).add(new THREE.Vector3(0, -0.06, -0.17));
    S.acc.backpack = { group: attach('Spine2', k), tint: [packM] };
  }

  // ---------------------------------------------------------------- per-frame
  function frame() {
    const dt = Math.min(S.clock.getDelta(), 0.1);
    S.time += dt;
    if (S.ready) {
      animateBody(dt);
      animateFace(dt);
      for (const U of Object.values(S.uniforms)) {
        U.uTime.value = S.time;
        U.uWind.value.copy(S.windDir).multiplyScalar(S.wind);
      }
      if (S.cloth) {                       // fixed 60 Hz physics, at most 3 sub-steps per frame
        S.acc60 = Math.min(S.acc60 + dt, 3 / 60);
        let stepped = false;
        while (S.acc60 >= 1 / 60) { S.cloth.step(1 / 60); S.acc60 -= 1 / 60; stepped = true; }
        if (stepped) S.cloth.upload();
      }
    }
    S.controls.update();
    S.renderer.render(S.scene, S.camera);
  }

  // one natural standing stance: arms relaxed down (opened a little over skirts), breathing
  function animateBody(dt) {
    const B = S.bones;
    for (const [name, r] of Object.entries(S.rest)) B[name].quaternion.copy(r.q);
    S.armOut += (S.armOutTarget - S.armOut) * Math.min(1, dt * 3);
    rotWorld(B.LeftArm, AX.z, S.armOut - S.bind.armAngL);
    rotWorld(B.RightArm, AX.z, -(S.armOut - S.bind.armAngR));
    // palms toward the thighs
    rotWorld(B.LeftArm, worldPos('LeftForeArm').sub(worldPos('LeftArm')).normalize(), 0.9);
    rotWorld(B.RightArm, worldPos('RightForeArm').sub(worldPos('RightArm')).normalize(), -0.9);
    rotWorld(B.LeftForeArm, AX.x, -0.18);
    rotWorld(B.RightForeArm, AX.x, -0.18);
    const breathe = Math.sin(S.time * 1.5);
    rotWorld(B.Spine1, AX.x, breathe * 0.01);
    rotWorld(B.Spine2, AX.x, breathe * 0.008);
  }

  function morph(name, v) {
    for (const m of ['Wolf3D_Head', 'EyeLeft', 'EyeRight', 'Wolf3D_Teeth']) {
      const mesh = S.meshes[m];
      if (mesh && mesh.morphTargetDictionary && name in mesh.morphTargetDictionary) mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = v;
    }
  }
  function animateFace(dt) {
    S.blinkT -= dt;
    let blink = 0;
    if (S.blinkT < 0.12) blink = Math.sin(Math.max(0, S.blinkT) / 0.12 * Math.PI);
    if (S.blinkT < 0) S.blinkT = 2 + Math.random() * 3.5;
    morph('eyeBlinkLeft', blink); morph('eyeBlinkRight', blink);
    S.smileT = Math.max(0, S.smileT - dt);
    morph('mouthSmile', Math.min(1, S.smileT) * 0.5);
  }

  // ---------------------------------------------------------------- body shaping
  function resetBones() {
    for (const [name, r] of Object.entries(S.rest)) { const b = S.bones[name]; b.position.copy(r.p); b.scale.copy(r.s); }
  }
  function scaleAxis(bone, worldAxis, k) {
    if (!bone || !isFinite(k) || Math.abs(k - 1) < 1e-3) return;
    bone.scale[AXN[localAxisFor(bone, worldAxis)]] *= k;
  }
  function headsTall(age) {           // stature in head-heights by age
    if (age == null || age >= 16) return 7.5;
    const pts = [[1, 4.0], [3, 5.0], [6, 6.0], [10, 6.6], [13, 7.0], [16, 7.5]];
    for (let i = 1; i < pts.length; i++) if (age <= pts[i][0]) {
      const [a0, h0] = pts[i - 1], [a1, h1] = pts[i];
      return h0 + (h1 - h0) * (age - a0) / (a1 - a0);
    }
    return 7.5;
  }

  function setMeasurements(m, info = {}) {
    if (!S.ready || !m) return;
    resetBones();
    for (const [name, r] of Object.entries(S.rest)) S.bones[name].quaternion.copy(r.q);   // shape in bind pose
    S.avatar.scale.setScalar(1);
    S.avatar.position.set(0, 0, 0);
    S.avatar.updateMatrixWorld(true);
    const h = (m.height_cm || 175) / 100;
    const s0 = h / S.bind.height;
    const B = S.bones;
    const kShoulder = clamp(((m.shoulder_cm || 0.26 * h * 100) / 100 / s0) / S.bind.shoulder, 0.75, 1.3);
    const kChest = clamp(((m.chest_cm || 0.55 * h * 100) / (h * 100)) / 0.55, 0.8, 1.35);
    const kWaist = clamp(((m.waist_cm || 0.45 * h * 100) / (h * 100)) / 0.45, 0.78, 1.45);
    const kHip = clamp(((m.hip_cm || 0.54 * h * 100) / (h * 100)) / 0.54, 0.8, 1.35);
    const kLeg = clamp(m.leg_cm ? (m.leg_cm / 100 / s0) / S.bind.leg : 1, 0.88, 1.12);
    const kArm = clamp(m.arm_cm ? (m.arm_cm / 100 / s0) / S.bind.arm : 1, 0.88, 1.12);
    const kHead = clamp(7.5 / headsTall(info.age), 1, 1.9);
    const child = info.age != null && info.age < 14;

    scaleAxis(B.Spine, AX.x, kWaist); scaleAxis(B.Spine, AX.z, kWaist);
    scaleAxis(B.Spine1, AX.x, kChest / kWaist); scaleAxis(B.Spine1, AX.z, kChest / kWaist);
    scaleAxis(B.Spine2, AX.x, kShoulder / kChest);
    scaleAxis(B.Neck, AX.x, 1 / kShoulder); scaleAxis(B.Neck, AX.z, 1 / kChest);   // keep the head round
    for (const side of ['Left', 'Right']) {
      scaleAxis(B[side + 'Shoulder'], AX.z, 1 / kChest);
      const p = B[side + 'UpLeg'].position, ax = localAxisFor(B.Hips, AX.x);
      p.setComponent(ax, S.rest[side + 'UpLeg'].p.getComponent(ax) * kHip);
      scaleAxis(B[side + 'UpLeg'], AX.x, kHip); scaleAxis(B[side + 'UpLeg'], AX.z, kHip);
      scaleAxis(B[side + 'Leg'], AX.x, 1 / kHip); scaleAxis(B[side + 'Leg'], AX.z, 1 / kHip);
      B[side + 'UpLeg'].scale.y *= kLeg; B[side + 'Foot'].scale.y /= kLeg;     // bones point along local Y
      B[side + 'Arm'].scale.y *= kArm; B[side + 'Hand'].scale.y /= kArm;
    }
    if (kHead > 1.01) {
      B.Head.scale.multiplyScalar(kHead);
      for (const side of ['Left', 'Right']) B[side + 'Hand'].scale.multiplyScalar(1 / Math.sqrt(kHead));
      if (child) B.Neck.scale.y *= 0.85;
    }
    S.body.hipK = kHip;
    S.avatar.updateMatrixWorld(true);
    const shapedH = worldPos('HeadTop_End').y - Math.min(worldPos('LeftToe_End').y, worldPos('RightToe_End').y) + S.bind.toeY;
    S.body.s = h / shapedH;
    S.avatar.scale.setScalar(S.body.s);
    S.avatar.position.y = 0;
    S.avatar.updateMatrixWorld(true);
    S.avatar.position.y = S.bind.toeY * S.body.s - Math.min(worldPos('LeftToe_End').y, worldPos('RightToe_End').y);
    S.avatar.updateMatrixWorld(true);
    animateBody(0);
    render();                                   // the cloth is rebuilt for the new body
    S.onStatus(`🎯 Avatar shaped to your measurements: ${Math.round(h * 100)} cm` + (child ? ' (child proportions)' : ''));
    frameCamera();
  }

  function setPhysique(mode) {
    if (!S.ready) return;
    const p = { athletic: { s: 1, w: 1 }, muscular: { s: 1.12, w: 1.06 }, slim: { s: 0.92, w: 0.88 } }[mode] || { s: 1, w: 1 };
    const h = S.bind.height * 100 * 0.985;
    setMeasurements({ height_cm: h, shoulder_cm: S.bind.shoulder * 100 * p.s, chest_cm: 0.55 * h * p.s,
      waist_cm: 0.45 * h * p.w, hip_cm: 0.54 * h * p.w });
    S.onStatus(`Body: ${mode}`);
  }

  // ---------------------------------------------------------------- outfit model
  // items without a known category (e.g. older saved outfits) are placed by their name
  function inferCategory(t) {
    if (/sunglass|watch|\bcaps?\b|\bhat\b|belt|bag|backpack|tote|clutch/.test(t)) return 'accessories';
    if (/shoe|sneaker|boot|loafer|sandal|heel|flip flop|trainer|runner/.test(t)) return 'footwear';
    if (/dress|saree|gown/.test(t)) return 'dresses';
    if (/jacket|blazer|coat|parka/.test(t)) return 'outerwear';
    if (/jean|trouser|chino|pant|short|skirt|legging|capri|cargo/.test(t)) return 'bottoms';
    if (/shirt|tee|t-shirt|top|hoodie|sweater|sweatshirt|kurta|polo|blouse/.test(t)) return 'tops';
    return null;
  }

  function slotFor(it) {
    const t = `${it.article_type || ''} ${it.name || ''}`.toLowerCase();
    const known = ['tops', 'outerwear', 'bottoms', 'dresses', 'footwear', 'accessories'];
    if (!known.includes(it.category)) {
      const cat = inferCategory(t);
      if (!cat) return null;
      it = { ...it, category: cat };
    }
    if (it.category === 'dresses' || /\b(dress|saree|romper|jumpsuit)/.test(t)) return 'dress';
    if (it.category === 'outerwear' || /\b(jacket|blazer|coat|waistcoat|shrug)/.test(t)) return 'outer';
    if (it.category === 'tops') return 'top';
    if (it.category === 'bottoms') return 'bottom';
    if (it.category === 'footwear') return 'shoes';
    if (it.category === 'accessories') {
      if (/sunglass|eyewear|spectacle/.test(t)) return 'sunglasses';
      if (/watch/.test(t)) return 'watch';
      if (/\bcap\b|\bcaps\b|hat/.test(t)) return 'cap';
      if (/belt/.test(t)) return 'belt';
      if (/backpack/.test(t)) return 'backpack';
      if (/bag|clutch|tote|purse/.test(t)) return 'bag';
    }
    return null;
  }

  const kindOf = (it) => {
    const t = `${it.article_type || ''} ${it.name || ''}`.toLowerCase();
    return {
      skirt: /skirt/.test(t), shorts: /short/.test(t), capri: /capri|3\/4/.test(t), kurta: /kurta|kurti|tunic/.test(t),
      shortSleeve: /tshirt|t-shirt|\btee\b|polo|\btops?\b|sleeveless|tank/.test(t), sleeveless: /sleeveless|tank|strap/.test(t),
      long: /maxi|gown|saree|long/.test(t), shirt: /shirt/.test(t) && !/t-shirt|tshirt|sweatshirt/.test(t),
      knit: /sweater|sweatshirt|hoodie|cardigan/.test(t),
    };
  };

  // dressing rules; returns the list of slots that were taken off
  function applyItem(it) {
    if (!S.ready || !it) return [];
    const slot = slotFor(it);
    if (!slot) { S.onStatus(`${it.name}: no 3D preview for this item yet`); return []; }
    const removed = [];
    const take = (s) => { if (S.outfit[s] && !S.outfit[s].base) removed.push(S.outfit[s].name); delete S.outfit[s]; };
    const item = { ...it, slot, category: ['tops', 'outerwear', 'bottoms', 'dresses', 'footwear', 'accessories'].includes(it.category)
      ? it.category : inferCategory(`${it.article_type || ''} ${it.name || ''}`.toLowerCase()) };
    if (slot === 'dress') { take('top'); take('bottom'); }
    if (slot === 'top' || slot === 'bottom') take('dress');
    if (slot === 'bag') take('backpack');          // one bag at a time
    if (slot === 'backpack') take('bag');
    take(slot);
    S.outfit[slot] = item;
    S.smileT = 1.2;
    render();
    S.onStatus(`Wearing: ${it.name}${it.fabric ? ' · ' + it.fabric : ''}` + (removed.length ? ` (took off ${removed.join(', ')})` : ''));
    return removed;
  }

  function removeSlot(slot) {
    if (!S.outfit[slot]) return;
    delete S.outfit[slot];
    render();
  }

  function resetOutfit(silent) {
    S.outfit = { top: DEFAULT_TOP, bottom: DEFAULT_BOTTOM, shoes: DEFAULT_SHOES };
    S.fit = null;
    render();
    if (!silent) S.onStatus('🧍 Outfit reset');
  }

  function wornItems() {
    return Object.entries(S.outfit).filter(([, v]) => v && !v.base).map(([slot, v]) => ({ slot, ...v }));
  }

  function paint(mesh, color, fabric, keepNormal = true) {
    const f = fabricOf(fabric);
    mesh.visible = true;
    mesh.material.map = null;
    mesh.material.normalMap = keepNormal ? mesh.userData.normalMap : null;
    mesh.material.color.copy(lin(color || '#4f46e5'));
    mesh.material.roughness = f.rough;
    mesh.material.metalness = f.metal;
    mesh.material.needsUpdate = true;
    return f;
  }

  function skin() { return SKIN_TONES[S.skinIndex].skin; }

  // rebuild every visual from S.outfit
  function render() {
    if (!S.ready) return;
    const O = S.outfit, top = S.uniforms.top, bot = S.uniforms.bottom, out = S.uniforms.outer;
    const topMesh = S.meshes.Wolf3D_Outfit_Top, botMesh = S.meshes.Wolf3D_Outfit_Bottom;
    for (const U of [top, bot, out]) {
      U.uCutY.value = -10; U.uSleeveX.value = 10; U.uShrink.value = 0; U.uHideAbove.value = 10; U.uHideBelow.value = -10;
      U.uBeltOn.value = 0;
      U.uSkin.value.copy(lin(skin()));
    }
    let cloth = null;

    // upper body
    if (O.dress) {
      const k = kindOf(O.dress);
      top.uSway.value = paint(topMesh, O.dress.color, O.dress.fabric, false).sway;
      top.uSleeveX.value = k.sleeveless ? 0.19 : 0.22;
      top.uHideBelow.value = S.bind.waistY - 0.01;         // the bodice ends where the skirt cloth starts
      cloth = { color: O.dress.color, fabric: O.dress.fabric, length: k.long ? 0.95 : 0.55, flare: 0.55, kind: 'dress' };
    } else {
      const t = O.top || DEFAULT_TOP, k = kindOf(t);
      top.uSway.value = paint(topMesh, t.color, t.fabric).sway;
      if (k.shortSleeve && !t.base) top.uSleeveX.value = k.sleeveless ? 0.19 : 0.27;
      if (k.kurta) {
        top.uHideBelow.value = S.bind.waistY - 0.01;
        cloth = { color: t.color, fabric: t.fabric, length: 0.42, flare: 0.25, kind: 'kurta' };
      }
    }

    // lower body
    const b = O.dress ? null : (O.bottom || DEFAULT_BOTTOM);
    const bk = b ? kindOf(b) : {};
    if (!b || bk.skirt) {
      // bare legs: the rig has no leg skin, so the trouser mesh is painted skin and slimmed
      paint(botMesh, skin(), 'cotton', false);
      botMesh.material.roughness = 0.6;
      bot.uShrink.value = -0.022;
      bot.uHideAbove.value = S.bind.hipsY - 0.06;
      bot.uSway.value = 0;
      if (bk.skirt) {
        cloth = { color: b.color, fabric: b.fabric, length: 0.45, flare: 0.6, kind: 'skirt' };
        top.uHideBelow.value = S.bind.waistY - 0.01;        // the top is tucked into the skirt
      }
    } else {
      bot.uSway.value = paint(botMesh, b.color, b.fabric).sway;
      bot.uCutY.value = bk.shorts ? 0.56 : bk.capri ? 0.3 : -10;
      if (cloth && cloth.kind === 'kurta') bot.uHideAbove.value = S.bind.hipsY - 0.02;   // waistband under the hem
      // shirts are tucked in when a belt is worn or the trousers are formal; the waistband then shows
      const t = O.dress ? null : (O.top || DEFAULT_TOP), tk = t ? kindOf(t) : {};
      const formal = /formal|trouser|chino/i.test(`${b.usage || ''} ${b.article_type || ''}`) && /formal/i.test(`${t && t.usage || ''}`);
      if (t && !tk.kurta && !tk.knit && (O.belt || (tk.shirt && formal))) top.uHideBelow.value = 1.072;
      // belt: recolour the belt that is modelled on the trousers
      bot.uBeltOn.value = O.belt ? 1 : 0;
      if (O.belt) bot.uBelt.value.copy(lin(O.belt.color || '#3f2a1d'));
    }

    // outer layer (jacket / blazer) over whatever is on top
    if (O.outer) {
      out.uSway.value = paint(S.outer, O.outer.color, O.outer.fabric).sway * 0.5;
      S.outer.visible = true;
      if (cloth && cloth.kind !== 'kurta') out.uHideBelow.value = S.bind.waistY - 0.01;
    } else S.outer.visible = false;

    // shoes
    paint(S.meshes.Wolf3D_Outfit_Footwear, (O.shoes || DEFAULT_SHOES).color, (O.shoes || DEFAULT_SHOES).fabric);

    // accessories
    for (const [slot, a] of Object.entries(S.acc)) {
      const it = O[slot];
      a.group.visible = !!it;
      if (it && it.color) a.tint.forEach((m) => { m.color.copy(lin(it.color)); m.needsUpdate = true; });
    }

    // cloth: rebuild only when the spec changed (or the body changed)
    const key = cloth ? JSON.stringify(cloth) + S.body.s + S.body.hipK : null;
    if (key !== S.clothKey) {
      if (S.cloth) { S.cloth.dispose(); S.cloth = null; }
      if (cloth) S.cloth = new Cloth(cloth);
      S.clothKey = key;
    }
    // arms open a little over skirts / with a jacket so hands and sleeves stay clear
    S.armOutTarget = cloth ? (cloth.kind === 'kurta' ? 0.36 : 0.4) : O.outer ? 0.18 : O.bag ? 0.2 : 0.13;
    applyFit();
    S.onOutfit(wornItems());
  }

  // ---------------------------------------------------------------- fit visualisation
  const VERDICT_COLOUR = { tight: '#ef4444', snug: '#f59e0b', good: '#22c55e', relaxed: '#38bdf8', loose: '#6366f1' };
  const IDEAL_EASE = { chest: 10, waist: 6, hip: 7 };

  function setFit(category, regions) { S.fit = regions ? { category, regions } : null; applyFit(); }
  function applyFit() {
    for (const U of Object.values(S.uniforms)) { U.uRegOn.value.set(0, 0, 0); U.uEase.value.set(0, 0, 0); }
    if (!S.fit) return;
    const { category, regions } = S.fit;
    const targets = category === 'outerwear' ? ['outer'] : category === 'bottoms' ? ['bottom'] : category === 'dresses' ? ['top'] : ['top'];
    const idx = { chest: 0, waist: 1, hip: 2 };
    for (const k of targets) {
      const U = S.uniforms[k];
      const on = [0, 0, 0], ease = [0, 0, 0], cols = [U.uHeatA.value, U.uHeatB.value, U.uHeatC.value];
      for (const r of regions) {
        const i = idx[r.region]; if (i == null) continue;
        on[i] = 1;
        // circumference ease -> radial offset in metres, x2 so it is visible; shrinking is capped so the
        // garment never sinks into the body
        ease[i] = clamp(((r.ease_cm - (IDEAL_EASE[r.region] || 6)) / (2 * Math.PI) / 100) * 2, -0.005, 0.025);
        cols[i].copy(lin(VERDICT_COLOUR[r.verdict] || '#22c55e'));
      }
      U.uRegOn.value.set(...on);
      U.uEase.value.set(...ease);
    }
  }
  function showHeat(on) { S.heat = !!on; for (const U of Object.values(S.uniforms)) U.uHeatOn.value = on ? 1 : 0; }

  // ---------------------------------------------------------------- misc controls
  function setWind(v) { S.wind = clamp(Number(v) || 0, 0, 1.5); }
  function cycleSkin() {
    S.skinIndex = (S.skinIndex + 1) % SKIN_TONES.length;
    const tone = SKIN_TONES[S.skinIndex];
    for (const n of ['Wolf3D_Head', 'Wolf3D_Body']) {
      const m = S.meshes[n]; if (m) { m.material.color.copy(lin(tone.tint)); m.material.needsUpdate = true; }
    }
    render();
    S.onStatus(`Skin tone: ${tone.name}`);
  }
  function toggleTop() { const m = S.meshes.Wolf3D_Outfit_Top; m.visible = !m.visible; }
  function frameCamera(view = S.view || 'full') {
    S.view = view;
    const h = S.bind.height * (S.body.s || 1);
    const target = { full: 0.5, upper: 0.74, face: 0.92 }[view] * h;
    const dist = { full: 1.8, upper: 0.85, face: 0.42 }[view] * Math.max(h, 1.0);
    S.controls.target.set(0, target, 0);
    S.camera.position.set(dist * 0.25, target + (view === 'full' ? 0.15 : 0.05) * h, dist);
    S.controls.update();
  }
  function spin(on) { S.controls.autoRotate = on; S.controls.autoRotateSpeed = 2.0; }

  // fast-forward body + cloth (screenshots / tests)
  function settle(seconds = 2) {
    if (!S.ready) return;
    for (let i = 0; i < seconds * 60; i++) { S.time += 1 / 60; animateBody(1 / 60); if (S.cloth) S.cloth.step(1 / 60); }
    if (S.cloth) S.cloth.upload();
  }

  // penetration check: how many cloth points sit inside a collider (should be 0)
  function clothPenetration() {
    if (!S.cloth) return 0;
    const col = S.cloth.colliders();
    let bad = 0;
    const p = S.cloth.p, n = p.length / 3;
    for (let i = S.cloth.cols; i < n; i++) {
      _v.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]);
      for (const c of col.caps) {
        _v2.subVectors(c.b, c.a);
        const t = clamp(_v3.subVectors(_v, c.a).dot(_v2) / _v2.lengthSq(), 0, 1);
        if (_v.distanceTo(_v3.copy(c.a).addScaledVector(_v2, t)) < c.r * 0.9) { bad++; break; }
      }
    }
    return bad;
  }

  window.AvatarStudio = {
    init, settle, applyItem, removeSlot, resetOutfit, wornItems, slotFor, setMeasurements, setPhysique, setWind,
    setFit, showHeat, cycleSkin, toggleTop, frameCamera, spin, clothPenetration, FABRICS,
    get ready() { return S.ready; }, _state: S,
  };
})();

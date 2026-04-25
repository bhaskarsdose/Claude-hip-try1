import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";

const fileInput = document.getElementById("file");
const drop = document.getElementById("drop");
const runBtn = document.getElementById("run");
const statusEl = document.getElementById("status");
const banner = document.getElementById("model-banner");
const dlInput = document.getElementById("dl-input");
const dlImplant = document.getElementById("dl-implant");

const inputViewer = createViewer(document.getElementById("viewer-input"), 0xb8c0cc);
const implantViewer = createViewer(document.getElementById("viewer-implant"), 0x6aa6ff);

let chosenFile = null;

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("hover"); });
drop.addEventListener("dragleave", () => drop.classList.remove("hover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("hover");
  if (e.dataTransfer.files.length) {
    chooseFile(e.dataTransfer.files[0]);
  }
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) chooseFile(fileInput.files[0]);
});
runBtn.addEventListener("click", run);

function chooseFile(f) {
  if (!/\.nii(\.gz)?$/i.test(f.name)) {
    setStatus(`Unsupported file: ${f.name}. Need .nii or .nii.gz`, true);
    return;
  }
  chosenFile = f;
  setStatus(`Selected: ${f.name} (${(f.size / 1024 / 1024).toFixed(1)} MB)`);
  runBtn.disabled = false;
}

async function run() {
  if (!chosenFile) return;
  runBtn.disabled = true;
  banner.classList.add("hidden");
  dlInput.classList.add("hidden");
  dlImplant.classList.add("hidden");
  inputViewer.clear();
  implantViewer.clear();
  setStatus("Uploading…");

  const fd = new FormData();
  fd.append("file", chosenFile);
  let resp;
  try {
    resp = await fetch("/api/reconstruct", { method: "POST", body: fd });
  } catch (e) {
    setStatus(`Network error: ${e}`, true);
    runBtn.disabled = false;
    return;
  }
  if (!resp.ok) {
    setStatus(`Upload failed: ${resp.status} ${await resp.text()}`, true);
    runBtn.disabled = false;
    return;
  }
  const { job_id } = await resp.json();
  setStatus(`Job ${job_id} queued. Reconstructing…`);
  pollJob(job_id);
}

async function pollJob(id) {
  for (let i = 0; i < 600; i++) {
    await sleep(1000);
    const r = await fetch(`/api/jobs/${id}`);
    if (!r.ok) {
      setStatus(`Job lookup failed: ${r.status}`, true);
      runBtn.disabled = false;
      return;
    }
    const j = await r.json();
    if (j.status === "done") {
      setStatus("Done.");
      showBanner(j.used_trained_model);
      await loadStl(`/api/jobs/${id}/input.stl`, inputViewer, dlInput, id, "input");
      await loadStl(`/api/jobs/${id}/implant.stl`, implantViewer, dlImplant, id, "implant");
      runBtn.disabled = false;
      return;
    }
    if (j.status === "failed") {
      setStatus(`Failed: ${j.error || "unknown error"}`, true);
      runBtn.disabled = false;
      return;
    }
    setStatus(`Status: ${j.status}…`);
  }
  setStatus("Timed out waiting for job.", true);
  runBtn.disabled = false;
}

function showBanner(usedTrained) {
  banner.classList.remove("hidden", "warn", "ok");
  if (usedTrained) {
    banner.classList.add("ok");
    banner.textContent = "Trained model used.";
  } else {
    banner.classList.add("warn");
    banner.textContent =
      "No trained checkpoint found in models/unet3d_hip.pt — running placeholder mode " +
      "(input is echoed, implant will be empty). Train via notebooks/train_colab.ipynb.";
  }
}

async function loadStl(url, viewer, dlEl, id, kind) {
  const resp = await fetch(url);
  if (!resp.ok) return;
  const buf = await resp.arrayBuffer();
  if (buf.byteLength === 0) return;  // empty placeholder STL
  const loader = new STLLoader();
  const geom = loader.parse(buf);
  geom.computeVertexNormals();
  viewer.setGeometry(geom);
  dlEl.href = url;
  dlEl.download = `${kind}_${id}.stl`;
  dlEl.classList.remove("hidden");
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("err", isError);
}

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

function createViewer(el, color) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  const resize = () => {
    const w = el.clientWidth, h = el.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0d12);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  camera.position.set(220, 180, 220);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dir = new THREE.DirectionalLight(0xffffff, 0.85);
  dir.position.set(1, 1, 1);
  scene.add(dir);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  let mesh = null;
  function setGeometry(geom) {
    if (mesh) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    geom.computeBoundingSphere();
    const center = geom.boundingSphere.center.clone();
    geom.translate(-center.x, -center.y, -center.z);
    const mat = new THREE.MeshStandardMaterial({
      color, metalness: 0.05, roughness: 0.6, flatShading: false,
    });
    mesh = new THREE.Mesh(geom, mat);
    scene.add(mesh);
    const r = geom.boundingSphere.radius || 100;
    camera.position.copy(new THREE.Vector3(1.2, 1.0, 1.6).multiplyScalar(r * 1.8));
    controls.target.set(0, 0, 0);
    controls.update();
  }

  function clear() {
    if (mesh) {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
      mesh = null;
    }
  }

  resize();
  new ResizeObserver(resize).observe(el);

  function loop() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  loop();

  return { setGeometry, clear };
}

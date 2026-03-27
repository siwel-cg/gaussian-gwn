import { PointCloud } from '../utils/load';
import bb_overlay_wgsl from '../shaders/bb-overlay.wgsl';
import gwn_compute_wgsl from '../shaders/gwn_compute.wgsl';
import precompute_wgsl from '../shaders/precompute.wgsl';
import { Renderer } from './renderer';
import { mat4, vec3 } from 'wgpu-matrix';
import orient_depth_wgsl from '../shaders/orient_depth.wgsl';
import orient_vote_wgsl from '../shaders/orient_vote.wgsl';

const GWN_WORKGROUP_SIZE = 64;

export interface BBRendererControls {
  setResolution:  (res: number) => void;
  setShowBBox:    (show: boolean) => void;
  setShowQuery:   (show: boolean) => void;
  setShowCameras: (show: boolean) => void;
  setGWNMode:     (enabled: boolean) => void;
  setPointSize:   (px: number) => void;
  runGWN:         () => void;
  setNumCameras:  (n: number) => void;
  setBounds:      (minX: number, minY: number, minZ: number,
                   maxX: number, maxY: number, maxZ: number) => void;
  getOriginalBounds: () => { min: number[], max: number[] };
}

export default function get_renderer_bb(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer
): Renderer & BBRendererControls {

  // ---- State ----
  let curMin     = [pc.bbox_min[0], pc.bbox_min[1], pc.bbox_min[2]];
  let curMax     = [pc.bbox_max[0], pc.bbox_max[1], pc.bbox_max[2]];
  let resolution = 10;
  let showBBox   = true;
  let showQuery  = true;
  let gwnMode    = false;
  let pointSize  = 4.0;
  let showCameras = false;

  function getPerAxisRes(): [number, number, number] {
    const dx = curMax[0] - curMin[0];
    const dy = curMax[1] - curMin[1];
    const dz = curMax[2] - curMin[2];
    const maxDim = Math.max(dx, dy, dz);
    if (maxDim === 0) return [1, 1, 1];
    const spacing = maxDim / resolution;
    return [
      Math.max(1, Math.round(dx / spacing)),
      Math.max(1, Math.round(dy / spacing)),
      Math.max(1, Math.round(dz / spacing)),
    ];
  }

  function totalQueryPoints() {
    const [rx, ry, rz] = getPerAxisRes();
    return rx * ry * rz;
  }

  // PRECOMPUTE PASS STUFF: PRE CALCS NORM AXIS, AREA 

  const PRECOMPUTED_STRIDE = 32; // 8 × f32
  const precomputed_buffer = device.createBuffer({
    label: 'precomputed splat data',
    size: pc.num_points * PRECOMPUTED_STRIDE,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const precompute_num_buf = device.createBuffer({
    label: 'precompute num_gaussians',
    size: 4,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(precompute_num_buf, 0, new Uint32Array([pc.num_points]));

  const precompute_pipeline = device.createComputePipeline({
    label: 'precompute-pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'precompute', code: precompute_wgsl }),
      entryPoint: 'precompute',
      constants: { workgroupSize: GWN_WORKGROUP_SIZE },
    },
  });

  const precompute_bg = device.createBindGroup({
    label: 'precompute-bg',
    layout: precompute_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: precomputed_buffer } },
      { binding: 2, resource: { buffer: precompute_num_buf } },
    ],
  });

  // RUN PRECOMPUTE
  function runPrecompute() {
    const dispatch = Math.ceil(pc.num_points / GWN_WORKGROUP_SIZE);
    const encoder = device.createCommandEncoder({ label: 'precompute' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(precompute_pipeline);
    pass.setBindGroup(0, precompute_bg);
    pass.dispatchWorkgroups(dispatch, 1, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }
  runPrecompute();

  // NORMAL CAMERA STUFF
  const CAMERA_SPHERE_SCALE = 1.5;
  const ORIENT_DEPTH_RES = 16;
  const ORIENT_FOV = Math.PI / 2;
  const ORIENT_NEAR = 0.01;
  const ORIENT_FAR_SCALE = 4.0;

  let numCameras = 16;

  const BYTES_PER_CAMERA = 80; // 20 floats

  function fibonacciSphere(n: number): [number, number, number][] {
    const pts: [number, number, number][] = [];
    const golden = (1 + Math.sqrt(5)) / 2;
    for (let i = 0; i < n; i++) {
      const theta = Math.acos(1 - 2 * (i + 0.5) / n);
      const phi   = 2 * Math.PI * i / golden;
      pts.push([
        Math.sin(theta) * Math.cos(phi),
        Math.sin(theta) * Math.sin(phi),
        Math.cos(theta),
      ]);
    }
    return pts;
  }

  function buildViewProj(eye: number[], target: number[], radius: number): Float32Array {
    const dir = vec3.normalize(vec3.subtract(target, eye));
    const dotUp = Math.abs(vec3.dot(dir, [0, 1, 0]));
    const safeUp = dotUp > 0.99 ? [1, 0, 0] : [0, 1, 0];

    const view = mat4.lookAt(eye, target, safeUp);
    const proj = mat4.perspective(ORIENT_FOV, 1.0, ORIENT_NEAR, ORIENT_FAR_SCALE * radius);
    return mat4.multiply(proj, view) as Float32Array;
  }

  let orient_camera_buffer: GPUBuffer;
  let orient_camera_count_buffer: GPUBuffer;

  function buildOrientCameras() {
    const cx = pc.centroid[0], cy = pc.centroid[1], cz = pc.centroid[2];
    const camRadius = CAMERA_SPHERE_SCALE * pc.radius;
    const dirs = fibonacciSphere(numCameras);

    const data = new Float32Array(numCameras * 20);
    for (let i = 0; i < numCameras; i++) {
      const [dx, dy, dz] = dirs[i];
      const eye = [cx + dx * camRadius, cy + dy * camRadius, cz + dz * camRadius];
      const target = [cx, cy, cz];
      const vp = buildViewProj(eye, target, pc.radius);

      const off = i * 20;
      data.set(vp, off);
      data[off + 16] = eye[0];
      data[off + 17] = eye[1];
      data[off + 18] = eye[2];
      data[off + 19] = 0;
    }

    if (orient_camera_buffer) orient_camera_buffer.destroy();
    orient_camera_buffer = device.createBuffer({
      label: 'orient cameras',
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(orient_camera_buffer, 0, data);

    if (orient_camera_count_buffer) orient_camera_count_buffer.destroy();
    orient_camera_count_buffer = device.createBuffer({
      label: 'orient camera count',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(orient_camera_count_buffer, 0,
      new Uint32Array([numCameras, ORIENT_DEPTH_RES, 0, 0]));

    console.log(`[bb-renderer] built ${numCameras} orientation cameras, radius=${camRadius.toFixed(3)}`);
  }

  buildOrientCameras();

  // ORIENT NORMALS
  const DEPTH_TOLERANCE = 0.0005;

  // Depth buffer: num_cameras * depth_res * depth_res u32s
  function depthBufSize() {
    return numCameras * ORIENT_DEPTH_RES * ORIENT_DEPTH_RES;
  }

  let orient_depth_buffer = device.createBuffer({
    label: 'orient depth buf',
    size: depthBufSize() * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // Params uniform: num_cameras, depth_res, num_splats, depth_tolerance
  const orient_params_buffer = device.createBuffer({
    label: 'orient params',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function writeOrientParams() {
    const buf = new ArrayBuffer(16);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u[0] = numCameras;
    u[1] = ORIENT_DEPTH_RES;
    u[2] = pc.num_points;
    f[3] = DEPTH_TOLERANCE;
    device.queue.writeBuffer(orient_params_buffer, 0, buf);
  }
  writeOrientParams();

  // Debug buffer for vote readback (vec4<f32> per splat: vote, in_frustum, depth_passes, grazing_skips)
  const vote_debug_buffer = device.createBuffer({
    label: 'vote debug',
    size: pc.num_points * 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const vote_debug_staging = device.createBuffer({
    label: 'vote debug staging',
    size: pc.num_points * 16,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // Depth buffer staging for readback
  let depth_debug_staging = device.createBuffer({
    label: 'depth debug staging',
    size: depthBufSize() * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  // --- Depth splat pipeline ---
  const orient_depth_pipeline = device.createComputePipeline({
    label: 'orient-depth-pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'orient-depth', code: orient_depth_wgsl }),
      entryPoint: 'depth_splat',
      constants: { workgroupSize: GWN_WORKGROUP_SIZE },
    },
  });

  // --- Vote pipeline ---
  const orient_vote_pipeline = device.createComputePipeline({
    label: 'orient-vote-pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'orient-vote', code: orient_vote_wgsl }),
      entryPoint: 'orient_vote',
      constants: { workgroupSize: GWN_WORKGROUP_SIZE },
    },
  });

  let orient_depth_bg: GPUBindGroup;
  let orient_vote_bg: GPUBindGroup;

  function rebuildOrientBindGroups() {
    orient_depth_bg = device.createBindGroup({
      label: 'orient-depth-bg',
      layout: orient_depth_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: precomputed_buffer } },
        { binding: 1, resource: { buffer: orient_camera_buffer } },
        { binding: 2, resource: { buffer: orient_params_buffer } },
        { binding: 3, resource: { buffer: orient_depth_buffer } },
      ],
    });

    orient_vote_bg = device.createBindGroup({
      label: 'orient-vote-bg',
      layout: orient_vote_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: precomputed_buffer } },
        { binding: 1, resource: { buffer: orient_camera_buffer } },
        { binding: 2, resource: { buffer: orient_params_buffer } },
        { binding: 3, resource: { buffer: orient_depth_buffer } },
        { binding: 4, resource: { buffer: vote_debug_buffer } },
      ],
    });
  }

  function runNormalOrientation() {
    // Clear depth buffer to 0xFFFFFFFF (max depth)
    const clearData = new Uint32Array(depthBufSize());
    clearData.fill(0xFFFFFFFF);
    device.queue.writeBuffer(orient_depth_buffer, 0, clearData);

    writeOrientParams();

    const dispatch = Math.ceil(pc.num_points / GWN_WORKGROUP_SIZE);
    const encoder = device.createCommandEncoder({ label: 'normal-orientation' });

    // Pass 1: depth splat
    const depthPass = encoder.beginComputePass();
    depthPass.setPipeline(orient_depth_pipeline);
    depthPass.setBindGroup(0, orient_depth_bg);
    depthPass.dispatchWorkgroups(dispatch, 1, 1);
    depthPass.end();

    // Pass 2: vote + flip
    const votePass = encoder.beginComputePass();
    votePass.setPipeline(orient_vote_pipeline);
    votePass.setBindGroup(0, orient_vote_bg);
    votePass.dispatchWorkgroups(dispatch, 1, 1);
    votePass.end();

    // Copy debug data for readback
    encoder.copyBufferToBuffer(vote_debug_buffer, 0, vote_debug_staging, 0, pc.num_points * 16);
    encoder.copyBufferToBuffer(orient_depth_buffer, 0, depth_debug_staging, 0, depthBufSize() * 4);

    device.queue.submit([encoder.finish()]);
    console.log(`[bb-renderer] normal orientation: ${numCameras} cameras, ${pc.num_points} splats`);

    // Async readback of vote debug data
    readbackOrientDebug();
  }

  async function readbackOrientDebug() {
    // --- Vote debug readback ---
    await vote_debug_staging.mapAsync(GPUMapMode.READ);
    const voteData = new Float32Array(vote_debug_staging.getMappedRange().slice(0));
    vote_debug_staging.unmap();

    let zeroFrustum = 0, zeroDepthPass = 0, flipped = 0;
    let totalVote = 0, totalFrustum = 0, totalDepthPass = 0, totalGrazing = 0;
    let minVote = Infinity, maxVote = -Infinity;

    for (let i = 0; i < pc.num_points; i++) {
      const vote       = voteData[i * 4 + 0];
      const inFrustum  = voteData[i * 4 + 1];
      const depthPasses = voteData[i * 4 + 2];
      const grazingSkips = voteData[i * 4 + 3];

      if (inFrustum === 0) zeroFrustum++;
      if (depthPasses === 0) zeroDepthPass++;
      if (vote < 0) flipped++;
      totalVote += vote;
      totalFrustum += inFrustum;
      totalDepthPass += depthPasses;
      totalGrazing += grazingSkips;
      minVote = Math.min(minVote, vote);
      maxVote = Math.max(maxVote, vote);
    }

    const n = pc.num_points;
    console.log(`[orient-debug] === Vote Pass Results ===`);
    console.log(`[orient-debug] Total splats: ${n}`);
    console.log(`[orient-debug] Splats with 0 cameras in frustum: ${zeroFrustum}`);
    console.log(`[orient-debug] Splats with 0 depth-test passes: ${zeroDepthPass}`);
    console.log(`[orient-debug] Splats flipped (vote < 0): ${flipped}`);
    console.log(`[orient-debug] Vote range: [${minVote.toFixed(4)}, ${maxVote.toFixed(4)}], avg: ${(totalVote / n).toFixed(4)}`);
    console.log(`[orient-debug] Avg cameras in frustum per splat: ${(totalFrustum / n).toFixed(1)}`);
    console.log(`[orient-debug] Avg cameras passing depth test: ${(totalDepthPass / n).toFixed(1)}`);
    console.log(`[orient-debug] Avg grazing skips per splat: ${(totalGrazing / n).toFixed(1)}`);

    // Log a few sample splats
    console.log(`[orient-debug] --- Sample splats (first 10) ---`);
    for (let i = 0; i < Math.min(10, n); i++) {
      const v = voteData[i*4], fr = voteData[i*4+1], dp = voteData[i*4+2], gs = voteData[i*4+3];
      console.log(`[orient-debug]   splat ${i}: vote=${v.toFixed(4)}, frustum=${fr}, depthPass=${dp}, grazing=${gs}`);
    }

    // --- Depth buffer readback ---
    await depth_debug_staging.mapAsync(GPUMapMode.READ);
    const depthData = new Uint32Array(depth_debug_staging.getMappedRange().slice(0));
    depth_debug_staging.unmap();

    const res = ORIENT_DEPTH_RES;
    const pixPerCam = res * res;
    console.log(`[orient-debug] === Per-Camera Depth Coverage ===`);
    for (let c = 0; c < numCameras; c++) {
      let written = 0, minD = 0xFFFFFFFF, maxD = 0;
      for (let p = 0; p < pixPerCam; p++) {
        const d = depthData[c * pixPerCam + p];
        if (d !== 0xFFFFFFFF) {
          written++;
          minD = Math.min(minD, d);
          maxD = Math.max(maxD, d);
        }
      }
      const coverage = (written / pixPerCam * 100).toFixed(1);
      const minNdc = (minD / 16777215).toFixed(4);
      const maxNdc = (maxD / 16777215).toFixed(4);
      console.log(`[orient-debug]   cam ${c}: ${written}/${pixPerCam} pixels (${coverage}%), depth NDC [${written > 0 ? minNdc : 'N/A'}, ${written > 0 ? maxNdc : 'N/A'}]`);
    }
  }

  // Rebuild bind groups and buffers when camera count changes
  function onCamerasChanged() {
    // Recreate depth buffer if camera count changed
    orient_depth_buffer.destroy();
    orient_depth_buffer = device.createBuffer({
      label: 'orient depth buf',
      size: depthBufSize() * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    // Recreate depth staging buffer to match new size
    depth_debug_staging.destroy();
    depth_debug_staging = device.createBuffer({
      label: 'depth debug staging',
      size: depthBufSize() * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    rebuildOrientBindGroups();
    rebuildCamVisBindGroups();
  }

  rebuildOrientBindGroups();

  // Run orientation immediately after precompute
  runNormalOrientation();
  
  // BB RENDER STUFF
  const UNIFORM_SIZE = 96;
  const uniformData  = new ArrayBuffer(UNIFORM_SIZE);
  const uf32 = new Float32Array(uniformData);
  const uu32 = new Uint32Array(uniformData);

  function writeUniforms() {
    const [rx, ry, rz] = getPerAxisRes();
    uf32[0]  = curMin[0]; uf32[1]  = curMin[1]; uf32[2]  = curMin[2]; uf32[3]  = 0;
    uf32[4]  = curMax[0]; uf32[5]  = curMax[1]; uf32[6]  = curMax[2]; uf32[7]  = 0;
    uu32[8]  = rx;
    uu32[9]  = ry;
    uu32[10] = rz;
    uu32[11] = showBBox ? 1 : 0;
    uu32[12] = showQuery ? 1 : 0;
    uu32[13] = gwnMode ? 1 : 0;
    uf32[14] = pointSize;
    uu32[15] = 0;
    uf32[16] = 0.0; uf32[17] = 1.0; uf32[18] = 1.0; uf32[19] = 1.0; // point color
    uf32[20] = 0.0; uf32[21] = 1.0; uf32[22] = 0.0; uf32[23] = 1.0; // box color
    device.queue.writeBuffer(bb_uniform_buffer, 0, uniformData);
  }

  const bb_uniform_buffer = device.createBuffer({
    label: 'bb overlay uniforms',
    size:  UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });


  // GNW CALC STUFF
  let maxQueryPoints = totalQueryPoints();
  let gwn_buffer = device.createBuffer({
    label: 'gwn values',
    size:  Math.max(4, maxQueryPoints * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const gwn_uniform_buffer = device.createBuffer({
    label: 'gwn uniforms', size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const gwn_bb_min_buffer = device.createBuffer({
    label: 'gwn bb_min', size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const gwn_bb_max_buffer = device.createBuffer({
    label: 'gwn bb_max', size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const gwn_grid_res_buffer = device.createBuffer({
    label: 'gwn grid_res', size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  function writeGWNComputeUniforms() {
    const [rx, ry, rz] = getPerAxisRes();
    const nq = rx * ry * rz;
    device.queue.writeBuffer(gwn_uniform_buffer,  0, new Uint32Array([pc.num_points, nq, 0, 0]));
    device.queue.writeBuffer(gwn_bb_min_buffer,   0, new Float32Array([curMin[0], curMin[1], curMin[2], 0]));
    device.queue.writeBuffer(gwn_bb_max_buffer,   0, new Float32Array([curMax[0], curMax[1], curMax[2], 0]));
    device.queue.writeBuffer(gwn_grid_res_buffer, 0, new Uint32Array([rx, ry, rz, 0]));
  }

  const compute_pipeline = device.createComputePipeline({
    label: 'gwn-compute-pipeline',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ label: 'gwn-compute', code: gwn_compute_wgsl }),
      entryPoint: 'compute_gwn',
      constants: { workgroupSize: GWN_WORKGROUP_SIZE },
    },
  });

  function makeComputeBindGroup() {
    return device.createBindGroup({
      label: 'gwn-compute-bg',
      layout: compute_pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: precomputed_buffer } },  // <-- changed
        { binding: 1, resource: { buffer: gwn_buffer } },
        { binding: 2, resource: { buffer: gwn_uniform_buffer } },
        { binding: 3, resource: { buffer: gwn_bb_min_buffer } },
        { binding: 4, resource: { buffer: gwn_bb_max_buffer } },
        { binding: 5, resource: { buffer: gwn_grid_res_buffer } },
      ],
    });
  }

  let compute_bg = makeComputeBindGroup();

  function ensureGWNBuffer() {
    const needed = totalQueryPoints();
    if (needed > maxQueryPoints) {
      gwn_buffer.destroy();
      maxQueryPoints = needed;
      gwn_buffer = device.createBuffer({
        label: 'gwn values',
        size:  maxQueryPoints * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      compute_bg = makeComputeBindGroup();
      rebuildRenderBindGroups();
    }
  }

  // RENDER SETUP
  const render_module = device.createShaderModule({ code: bb_overlay_wgsl });
  const bbox_pipeline = device.createRenderPipeline({
    label: 'bbox wireframe',
    layout: 'auto',
    vertex:   { module: render_module, entryPoint: 'vs_bbox' },
    fragment: { module: render_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
    primitive: { topology: 'line-list' },
  });

  const query_pipeline = device.createRenderPipeline({
    label: 'query point quads',
    layout: 'auto',
    vertex:   { module: render_module, entryPoint: 'vs_query' },
    fragment: { module: render_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
    primitive: { topology: 'triangle-list' },
  });

  // Camera visualization pipeline
  const cam_vis_pipeline = device.createRenderPipeline({
    label: 'camera vis points',
    layout: 'auto',
    vertex:   { module: render_module, entryPoint: 'vs_cameras' },
    fragment: { module: render_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
    primitive: { topology: 'triangle-list' },
  });

  let camera_bg_camvis: GPUBindGroup;   // group 0 - camera uniforms
  let orient_bg_camvis: GPUBindGroup;   // group 1 - orient camera data

  function rebuildCamVisBindGroups() {
    camera_bg_camvis = device.createBindGroup({
      layout: cam_vis_pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });
    orient_bg_camvis = device.createBindGroup({
      layout: cam_vis_pipeline.getBindGroupLayout(1),
      entries: [
        { binding: 1, resource: { buffer: orient_camera_buffer } },
      ],
    });
  }

  const camera_bg_bbox = device.createBindGroup({
    layout: bbox_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });
  const bb_bg_bbox = device.createBindGroup({
    layout: bbox_pipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: bb_uniform_buffer } }],
  });

  let camera_bg_query: GPUBindGroup;
  let bb_bg_query:     GPUBindGroup;
  let gwn_bg_query:    GPUBindGroup;

  function rebuildRenderBindGroups() {
    camera_bg_query = device.createBindGroup({
      layout: query_pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
    });
    bb_bg_query = device.createBindGroup({
      layout: query_pipeline.getBindGroupLayout(1),
      entries: [{ binding: 0, resource: { buffer: bb_uniform_buffer } }],
    });
    gwn_bg_query = device.createBindGroup({
      layout: query_pipeline.getBindGroupLayout(2),
      entries: [{ binding: 0, resource: { buffer: gwn_buffer } }],
    });
  }

  rebuildRenderBindGroups();
  rebuildCamVisBindGroups();
  writeUniforms();
  writeGWNComputeUniforms();

  function runGWN() {
    ensureGWNBuffer();
    writeGWNComputeUniforms();
    const dispatch_x = Math.ceil(totalQueryPoints() / GWN_WORKGROUP_SIZE);
    const encoder = device.createCommandEncoder({ label: 'gwn-compute' });
    const pass = encoder.beginComputePass();
    pass.setPipeline(compute_pipeline);
    pass.setBindGroup(0, compute_bg);
    pass.dispatchWorkgroups(dispatch_x, 1, 1);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  // RENDER
  function render(encoder: GPUCommandEncoder, texture_view: GPUTextureView) {
    if (!showBBox && !showQuery && !showCameras) return;

    const pass = encoder.beginRenderPass({
      label: 'bb overlay render',
      colorAttachments: [{ view: texture_view, loadOp: 'load', storeOp: 'store' }],
    });

    if (showBBox) {
      pass.setPipeline(bbox_pipeline);
      pass.setBindGroup(0, camera_bg_bbox);
      pass.setBindGroup(1, bb_bg_bbox);
      pass.draw(24);
    }

    if (showQuery) {
      const [rx, ry, rz] = getPerAxisRes();
      pass.setPipeline(query_pipeline);
      pass.setBindGroup(0, camera_bg_query);
      pass.setBindGroup(1, bb_bg_query);
      pass.setBindGroup(2, gwn_bg_query);
      pass.draw(rx * ry * rz * 6);
    }

    if (showCameras) {
      pass.setPipeline(cam_vis_pipeline);
      pass.setBindGroup(0, camera_bg_camvis);
      pass.setBindGroup(1, orient_bg_camvis);
      pass.draw(numCameras * 6);
    }

    pass.end();
  }

  return {
    frame: (encoder, texture_view) => render(encoder, texture_view),
    camera_buffer,

    setResolution(res) { resolution = res; writeUniforms(); },
    setShowBBox(show) { showBBox = show; writeUniforms(); },
    setShowQuery(show) { showQuery = show; writeUniforms(); },
    setShowCameras(show) { showCameras = show; },
    setGWNMode(enabled) { gwnMode = enabled; writeUniforms(); },
    setPointSize(px) { pointSize = px; writeUniforms(); },
    setNumCameras(n) {
        numCameras = n;
        runPrecompute();  // reset normals to fresh eigendecomposition before re-orienting
        buildOrientCameras();
        onCamerasChanged();
        runNormalOrientation();
      },
    runGWN,

    setBounds(minX, minY, minZ, maxX, maxY, maxZ) {
      curMin = [minX, minY, minZ];
      curMax = [maxX, maxY, maxZ];
      writeUniforms();
    },
    getOriginalBounds() {
      return {
        min: [pc.bbox_min[0], pc.bbox_min[1], pc.bbox_min[2]],
        max: [pc.bbox_max[0], pc.bbox_max[1], pc.bbox_max[2]],
      };
    },
  };
}
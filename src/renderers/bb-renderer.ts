import { PointCloud } from '../utils/load';
import bb_overlay_wgsl from '../shaders/bb-overlay.wgsl';
import gwn_compute_wgsl from '../shaders/gwn_compute.wgsl';
import { Renderer } from './renderer';

const GWN_WORKGROUP_SIZE = 64;

export interface BBRendererControls {
  setResolution:  (res: number)  => void;
  setShowBBox:    (show: boolean) => void;
  setShowQuery:   (show: boolean) => void;
  setGWNMode:     (enabled: boolean) => void;
  setPointSize:   (px: number)   => void;
  runGWN:         () => void;
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
  let pointSize  = 4.0; // pixels

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

  // ---- BB Uniform buffer ----
  // Layout (bytes 0-111, 7 × vec4):
  //  [0]  bb_min.xyz + pad
  //  [16] bb_max.xyz + pad
  //  [32] res_x, res_y, res_z, show_bbox   (u32 × 4)
  //  [48] show_query, gwn_mode, point_size(f32), pad
  //  [64] query_color (vec4)
  //  [80] bbox_color  (vec4)
  const UNIFORM_SIZE = 96;
  const uniformData  = new ArrayBuffer(UNIFORM_SIZE);
  const uf32 = new Float32Array(uniformData);
  const uu32 = new Uint32Array(uniformData);

  function writeUniforms() {
    const [rx, ry, rz] = getPerAxisRes();
    uf32[0]  = curMin[0]; uf32[1]  = curMin[1]; uf32[2]  = curMin[2]; uf32[3]  = 0;
    uf32[4]  = curMax[0]; uf32[5]  = curMax[1]; uf32[6]  = curMax[2]; uf32[7]  = 0;
    uu32[8]  = rx;        uu32[9]  = ry;        uu32[10] = rz;
    uu32[11] = showBBox  ? 1 : 0;
    uu32[12] = showQuery ? 1 : 0;
    uu32[13] = gwnMode   ? 1 : 0;
    uf32[14] = pointSize;           // point_size — f32 written into u32 slot 14
    uu32[15] = 0;
    uf32[16] = 0.0; uf32[17] = 1.0; uf32[18] = 1.0; uf32[19] = 1.0; // query_color cyan
    uf32[20] = 0.0; uf32[21] = 1.0; uf32[22] = 0.0; uf32[23] = 1.0; // bbox_color  green
    device.queue.writeBuffer(bb_uniform_buffer, 0, uniformData);
  }

  const bb_uniform_buffer = device.createBuffer({
    label: 'bb overlay uniforms',
    size:  UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // ---- GWN value buffer ----
  let maxQueryPoints = totalQueryPoints();

  let gwn_buffer = device.createBuffer({
    label: 'gwn values',
    size:  Math.max(4, maxQueryPoints * 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  // ---- GWN compute uniforms ----
  const gwn_uniform_buffer = device.createBuffer({
    label: 'gwn uniforms', size: 32,
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

  // ---- Compute pipeline ----
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
        { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
        { binding: 1, resource: { buffer: gwn_buffer            } },
        { binding: 2, resource: { buffer: gwn_uniform_buffer    } },
        { binding: 3, resource: { buffer: gwn_bb_min_buffer     } },
        { binding: 4, resource: { buffer: gwn_bb_max_buffer     } },
        { binding: 5, resource: { buffer: gwn_grid_res_buffer   } },
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

  // ---- Render pipelines ----
  const render_module = device.createShaderModule({ code: bb_overlay_wgsl });

  const bbox_pipeline = device.createRenderPipeline({
    label: 'bbox wireframe',
    layout: 'auto',
    vertex:    { module: render_module, entryPoint: 'vs_bbox' },
    fragment:  { module: render_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
    primitive: { topology: 'line-list' },
  });

  const query_pipeline = device.createRenderPipeline({
    label: 'query point quads',
    layout: 'auto',
    vertex:    { module: render_module, entryPoint: 'vs_query' },
    fragment:  { module: render_module, entryPoint: 'fs_main', targets: [{ format: presentation_format }] },
    primitive: { topology: 'triangle-list' },  // billboard quads = 6 verts each
  });

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
  writeUniforms();
  writeGWNComputeUniforms();

  // ---- GWN dispatch ----
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

  // ---- Render ----
  function render(encoder: GPUCommandEncoder, texture_view: GPUTextureView) {
    if (!showBBox && !showQuery) return;

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
      pass.draw(rx * ry * rz * 6); // 6 verts per billboard quad
    }

    pass.end();
  }

  return {
    frame: (encoder, texture_view) => render(encoder, texture_view),
    camera_buffer,

    setResolution(res)      { resolution = res;     writeUniforms(); },
    setShowBBox(show)       { showBBox   = show;    writeUniforms(); },
    setShowQuery(show)      { showQuery  = show;    writeUniforms(); },
    setGWNMode(enabled)     { gwnMode    = enabled; writeUniforms(); },
    setPointSize(px)        { pointSize  = px;      writeUniforms(); },
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
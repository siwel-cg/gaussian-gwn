import { PointCloud } from '../utils/load';
import bb_overlay_wgsl from '../shaders/bb-overlay.wgsl';
import { Renderer } from './renderer';

export interface BBRendererControls {
  setResolution: (res: number) => void;
  setShowBBox: (show: boolean) => void;
  setShowQuery: (show: boolean) => void;
}

export default function get_renderer_bb(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer
): Renderer & BBRendererControls {

  // --- Uniform data ---
  // Layout (std140):
  //  0: bb_min (vec3f) + pad
  // 16: bb_max (vec3f) + pad
  // 32: resolution (u32), show_bbox (u32), show_query (u32), pad
  // 48: query_color (vec4f)
  // 64: bbox_color (vec4f)
  // total: 80 bytes

  const UNIFORM_SIZE = 80;
  const uniformData = new ArrayBuffer(UNIFORM_SIZE);
  const f32 = new Float32Array(uniformData);
  const u32 = new Uint32Array(uniformData);

  let resolution = 10;
  let showBBox = true;
  let showQuery = true;

  function writeUniforms() {
    // bb_min
    f32[0] = pc.bbox_min[0];
    f32[1] = pc.bbox_min[1];
    f32[2] = pc.bbox_min[2];
    f32[3] = 0; // pad
    // bb_max
    f32[4] = pc.bbox_max[0];
    f32[5] = pc.bbox_max[1];
    f32[6] = pc.bbox_max[2];
    f32[7] = 0; // pad
    // resolution, show flags
    u32[8] = resolution;
    u32[9] = showBBox ? 1 : 0;
    u32[10] = showQuery ? 1 : 0;
    u32[11] = 0; // pad
    // query_color: cyan
    f32[12] = 0.0;
    f32[13] = 1.0;
    f32[14] = 1.0;
    f32[15] = 1.0;
    // bbox_color: green
    f32[16] = 0.0;
    f32[17] = 1.0;
    f32[18] = 0.0;
    f32[19] = 1.0;

    device.queue.writeBuffer(bb_uniform_buffer, 0, uniformData);
  }

  const bb_uniform_buffer = device.createBuffer({
    label: 'bb overlay uniforms',
    size: UNIFORM_SIZE,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const shader = device.createShaderModule({ code: bb_overlay_wgsl });

  // --- Bounding box pipeline (line-list) ---
  const bbox_pipeline = device.createRenderPipeline({
    label: 'bbox wireframe',
    layout: 'auto',
    vertex: {
      module: shader,
      entryPoint: 'vs_bbox',
    },
    fragment: {
      module: shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'line-list',
    },
  });

  // --- Query points pipeline (point-list) ---
  const query_pipeline = device.createRenderPipeline({
    label: 'query points',
    layout: 'auto',
    vertex: {
      module: shader,
      entryPoint: 'vs_query',
    },
    fragment: {
      module: shader,
      entryPoint: 'fs_main',
      targets: [{ format: presentation_format }],
    },
    primitive: {
      topology: 'point-list',
    },
  });

  // Bind groups - both pipelines share same layout since same shader bindings
  const bbox_camera_bg = device.createBindGroup({
    label: 'bbox camera',
    layout: bbox_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const bbox_bb_bg = device.createBindGroup({
    label: 'bbox uniforms',
    layout: bbox_pipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: bb_uniform_buffer } }],
  });

  const query_camera_bg = device.createBindGroup({
    label: 'query camera',
    layout: query_pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: camera_buffer } }],
  });

  const query_bb_bg = device.createBindGroup({
    label: 'query uniforms',
    layout: query_pipeline.getBindGroupLayout(1),
    entries: [{ binding: 0, resource: { buffer: bb_uniform_buffer } }],
  });

  // Write initial uniform values
  writeUniforms();

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    if (!showBBox && !showQuery) return;

    const pass = encoder.beginRenderPass({
      label: 'bb overlay render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'load',  // draw on top of existing content
          storeOp: 'store',
        },
      ],
    });

    if (showBBox) {
      pass.setPipeline(bbox_pipeline);
      pass.setBindGroup(0, bbox_camera_bg);
      pass.setBindGroup(1, bbox_bb_bg);
      pass.draw(24); // 12 edges * 2 vertices
    }

    if (showQuery) {
      pass.setPipeline(query_pipeline);
      pass.setBindGroup(0, query_camera_bg);
      pass.setBindGroup(1, query_bb_bg);
      pass.draw(resolution * resolution * resolution);
    }

    pass.end();
  };

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      render(encoder, texture_view);
    },
    camera_buffer,

    setResolution(res: number) {
      resolution = res;
      writeUniforms();
    },
    setShowBBox(show: boolean) {
      showBBox = show;
      writeUniforms();
    },
    setShowQuery(show: boolean) {
      showQuery = show;
      writeUniforms();
    },
  };
}
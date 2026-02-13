import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';

export interface GaussianRenderer extends Renderer {
  setGaussianMultiplier: (v: number) => void;
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================

  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: C.histogram_wg_size,
        sortKeyPerThread: c_histogram_block_rows,
      },
    },
  });

  const camera_bind_group_preprocess = device.createBindGroup({
    label: 'gauss camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

    // GAUSSIAN MULT BUFFER + UPDATE
  const paramsBuffer = device.createBuffer({
    label: 'params buffer',
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  function setGaussianMultiplier(gmult: number) {
    device.queue.writeBuffer(paramsBuffer, 0, new Float32Array([gmult]));
  }

  const gaussian_bind_group_preprocess = device.createBindGroup({
    label: 'point gaussians prepass',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }},
      {binding: 1, resource: { buffer: pc.sh_buffer }},
      {binding: 2, resource: { buffer: paramsBuffer}}
    ],
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });


  // ===============================================
  //    TODO: Create Render Pipeline and Bind Groups
  // =============================================== 

  const render_pipeline = device.createRenderPipeline({
    label: "gauss render pipeline",
    layout: 'auto',
    vertex: {
      module: device.createShaderModule({
        label: "gauss vertex shader",
        code: renderWGSL
      }),
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 8,
        attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x2' }
        ],
      }],
    },
    fragment: {
      module: device.createShaderModule({
        label: "gauss frag shader",
        code: renderWGSL
      }),
      entryPoint: 'fs_main', 
      targets: [{ 
        format: presentation_format,
        blend: {
          color: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
          alpha: {
            srcFactor: 'one',
            dstFactor: 'one-minus-src-alpha',
            operation: 'add',
          },
        },
      }]
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const camera_bind_group = device.createBindGroup({
    label: 'gauss camera',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [{binding: 0, resource: { buffer: camera_buffer }}],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'point gaussians',
    layout: render_pipeline.getBindGroupLayout(1),
    entries: [
      {binding: 0, resource: { buffer: pc.gaussian_3d_buffer }}
    ],
  });

  // WE MAKING THE QUAD SETUP HERE: \(^u^)/
  const quadOffset = new Float32Array([
    -1,-1,  1,-1,  1, 1,
    -1,-1,  1, 1, -1, 1,
  ]);

  const quad_buffer = device.createBuffer({
    label: 'quad buffer',
    size: 48,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
  });

  device.queue.writeBuffer(quad_buffer, 0, quadOffset);
  
  // SPLAT BUFFER FOR PREPROCESSING AND RENDERING
  const splat_buffer = device.createBuffer({
    label: 'splat buffer',
    size: pc.num_points * 48,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // FRUSTRUM INDEX BUFFER
  const splat_idx_buffer = device.createBuffer({
    label: 'splat index buffer',
    size: pc.num_points * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  // INDIRECT DRAW BUFFER
  const indirect_buffer = device.createBuffer({
    label: 'indirect buffer',
    size: 16,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE,
    mappedAtCreation: true
  });

  new Uint32Array(indirect_buffer.getMappedRange()).set([6, 0, 0, 0]);
  indirect_buffer.unmap();

  // BIND TO BOTH COMPUTE AND GAUSS SO YOU CAN USE IT IN BOTH
  const prepass_splat_bind_group = device.createBindGroup({
    label: 'preprocess splat bg',
    layout: preprocess_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: splat_idx_buffer}},
      { binding: 2, resource: { buffer: indirect_buffer}}
    ]
  });

  const render_splat_bind_group = device.createBindGroup({
    label: 'render splat bg',
    layout: render_pipeline.getBindGroupLayout(3),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: splat_idx_buffer}}
    ]
  });

  
  // FRUSTRUM INDICES BUFFERS
  // const prepass_splat_idx_bind_group = device.createBindGroup({
  //   label: 'prepass splat idx bg',
  //   layout: preprocess_pipeline.getBindGroupLayout(4),
  //   entries: [
  //     {binding: 0, resource: { buffer: splat_idx_buffer }}
  //   ]
  // });

  // const render_splat_idx_bind_group = device.createBindGroup({
  //   label: 'render splat idx bg',
  //   layout: render_pipeline.getBindGroupLayout(4),
  //   entries: [
  //     {binding: 0, resource: { buffer: splat_idx_buffer }}
  //   ]
  // });

  // BIND SORT BUFFERS TO RENDERER:
  const sort_bind_group_renderer = device.createBindGroup({
    label: 'renderer sort',
    layout: render_pipeline.getBindGroupLayout(2),
    entries: [
      // { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      // { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 0, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      // { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  // ===============================================
  //    TODO: Command Encoder Functions
  // ===============================================


  // const indirect_bind_group = device.createBindGroup({
  //   label: 'inirect bind group',
  //   layout: preprocess_pipeline.getBindGroupLayout(5),
  //   entries: [
  //     { binding: 0, resource: { buffer: indirect_buffer}}
  //   ]
  // });


  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gauss render',
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group_renderer);
    pass.setBindGroup(3, render_splat_bind_group);
    // pass.setBindGroup(4, render_splat_idx_bind_group);

    pass.setVertexBuffer(0, quad_buffer);


    // INDIRECT DRAW
    pass.drawIndirect(indirect_buffer, 0);

    // HARD CODED DRAW 
    //pass.draw(6, pc.num_points);

    pass.end();
  };

  // ===============================================
  //    TODO: Return Render Object
  // ===============================================

  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      encoder.clearBuffer(sorter.sort_info_buffer, 0, 4);

      const prepass = encoder.beginComputePass({
        label: 'prepass compute'
      });
      prepass.setPipeline(preprocess_pipeline);
      prepass.setBindGroup(0, camera_bind_group_preprocess);
      prepass.setBindGroup(1, gaussian_bind_group_preprocess);
      prepass.setBindGroup(2, sort_bind_group);
      prepass.setBindGroup(3, prepass_splat_bind_group);
      // prepass.setBindGroup(4, prepass_splat_idx_bind_group);
      // prepass.setBindGroup(5, indirect_bind_group);

      const numGroups = Math.ceil(pc.num_points / 256);
      prepass.dispatchWorkgroups(numGroups, 1, 1);
      prepass.dispatchWorkgroups(1, 1, 1);
      prepass.end();

      sorter.sort(encoder);

      render(encoder, texture_view);
    },
    camera_buffer,
    setGaussianMultiplier 
  };
}



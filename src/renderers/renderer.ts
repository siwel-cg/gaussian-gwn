import { load } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { default as get_renderer_bb, BBRendererControls } from './bb-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';


export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let bbRenderer: (Renderer & BBRendererControls) | undefined;
  let renderer: Renderer | undefined; 
  let cameras;
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
    show_bbox: true,
    show_query: true,
    grid_resolution: 10,
    num_cameras: 16,
    show_cameras: false,
    show_normals: false,
    normal_length: 0.05,
    bb_scale_x: 1.0,
    bb_scale_y: 1.0,
    bb_scale_z: 1.0,
    bb_offset_x: 0.0,
    bb_offset_y: 0.0,
    bb_offset_z: 0.0
  };

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });

  const overlay_folder = pane.addFolder({ title: 'Overlay' });
  overlay_folder.addInput(params, 'show_bbox', { label: 'Bounding Box' })
    .on('change', (e) => {
      bbRenderer?.setShowBBox(e.value);
    });

  overlay_folder.addInput(params, 'show_query', { label: 'Query Points' })
    .on('change', (e) => {
      bbRenderer?.setShowQuery(e.value);
    });

  overlay_folder.addInput(params, 'grid_resolution', {
    label: 'Grid Res',
    min: 2,
    max: 100,
    step: 1,
  })
  .on('change', (e) => {
    bbRenderer?.setResolution(e.value);
  });

  overlay_folder.addInput(params, 'show_cameras', { label: 'Show Cameras' })
    .on('change', (e) => {
      bbRenderer?.setShowCameras(e.value);
    });

  overlay_folder.addInput(params, 'show_normals', { label: 'Show Normals' })
    .on('change', (e) => {
      bbRenderer?.setShowNormals(e.value);
    });

  overlay_folder.addInput(params, 'normal_length', {
    label: 'Normal Len',
    min: 0.01,
    max: 0.5,
    step: 0.01,
  })
  .on('change', (e) => {
    bbRenderer?.setNormalLength(e.value);
  });

  overlay_folder.addInput(params, 'num_cameras', {
    label: 'Orient Cams',
    min: 4,
    max: 64,
    step: 2,
  })
  .on('change', (e) => {
    bbRenderer?.setNumCameras(e.value);
  });

  overlay_folder.addInput(params, 'bb_scale_x', { label: 'Scale X', min: 0.1, max: 3.0 })
    .on('change', () => updateBBTransform());
  overlay_folder.addInput(params, 'bb_scale_y', { label: 'Scale Y', min: 0.1, max: 3.0 })
    .on('change', () => updateBBTransform());
  overlay_folder.addInput(params, 'bb_scale_z', { label: 'Scale Z', min: 0.1, max: 3.0 })
    .on('change', () => updateBBTransform());
  overlay_folder.addInput(params, 'bb_offset_x', { label: 'Offset X', min: -5.0, max: 5.0 })
    .on('change', () => updateBBTransform());
  overlay_folder.addInput(params, 'bb_offset_y', { label: 'Offset Y', min: -5.0, max: 5.0 })
    .on('change', () => updateBBTransform());
  overlay_folder.addInput(params, 'bb_offset_z', { label: 'Offset Z', min: -5.0, max: 5.0 })
    .on('change', () => updateBBTransform());

  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const pc = await load(uploadedFile, device); // THIS WHERE WE LOAD FILE
        control.setTarget(pc.centroid, pc.radius * 2.5);
        camera.update_buffer();
        bbRenderer = get_renderer_bb(pc, device, presentation_format, camera.uniform_buffer);
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer);
        gaussian_renderer.setGaussianMultiplier(params.gaussian_multiplier);
        
        renderers = {
          pointcloud: pointcloud_renderer,
          gaussian: gaussian_renderer,
        };
        renderer = renderers[params.renderer];
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        //camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {

    pane.addInput(
      params,
      'gaussian_multiplier',
      {min: 0, max: 1.5}
    ).on('change', (e) => {
      gaussian_renderer.setGaussianMultiplier(e.value);
    });

  }

  // document.addEventListener('keydown', (event) => {
  //   switch(event.key) {
  //     case '0':
  //     case '1':
  //     case '2':
  //     case '3':
  //     case '4':
  //     case '5':
  //     case '6':
  //     case '7':
  //     case '8':
  //     case '9':
  //       const i = parseInt(event.key);
  //       console.log(`set to camera preset ${i}`);
  //       camera.set_preset(cameras[i]);
  //       break;
  //   }
  // });

  pane.addButton({ title: 'Compute GWN' }).on('click', () => {
      bbRenderer.setGWNMode(true);
      bbRenderer.runGWN();
  });

  function updateBBTransform() {
    if (!bbRenderer) return;
    const { min, max } = bbRenderer.getOriginalBounds();
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const hx = (max[0] - min[0]) / 2;
    const hy = (max[1] - min[1]) / 2;
    const hz = (max[2] - min[2]) / 2;

    bbRenderer.setBounds(
        cx + params.bb_offset_x - hx * params.bb_scale_x,
        cy + params.bb_offset_y - hy * params.bb_scale_y,
        cz + params.bb_offset_z - hz * params.bb_scale_z,
        cx + params.bb_offset_x + hx * params.bb_scale_x,
        cy + params.bb_offset_y + hy * params.bb_scale_y,
        cz + params.bb_offset_z + hz * params.bb_scale_z,
    );
  }

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      if (bbRenderer) {
          bbRenderer.frame(encoder, texture_view);
      }
      device.queue.submit([encoder.finish()]);
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}

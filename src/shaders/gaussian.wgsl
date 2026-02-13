struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec3<f32>,
    @location(2) conic: vec3<f32>,      
    @location(3) opacity: f32,
    //TODO: information passed from vertex shader to fragment shader
};

struct VertexInput {
    @location(0) corner: vec2<f32>
}

struct Splat {
    NDCpos: vec4<f32>,
    conic: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    opacity: f32
};


struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    _pad1: vec2<f32>,
    focal: vec2<f32>,
    _pad2: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}


@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians : array<Gaussian>;

// SORTS
// @group(2) @binding(0)
// var<storage, read_write> sort_infos: SortInfos;
// @group(2) @binding(1)
// var<storage, read> sort_depths : array<u32>;
@group(2) @binding(0)
var<storage, read> sort_indices : array<u32>;
// @group(2) @binding(3)
// var<storage, read_write> sort_dispatch: DispatchIndirect;

// SPLATS
@group(3) @binding(0)
var<storage, read> splatList : array<Splat>;
@group(3) @binding(1)
var<storage, read> splatIndexList : array<u32>;

@vertex
fn vs_main(in : VertexInput, @builtin(instance_index) instance: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;
    let vertex = gaussians[instance];

    let sorted_culled_idx = sort_indices[instance];
    let gaussian_idx  = splatIndexList[sorted_culled_idx];
    let splat = splatList[gaussian_idx];
    let rad = splat.radius;

    let clipPos = splat.NDCpos;// camera.proj * camera.view *  pos; //

    let px2ndc = vec2<f32>(2.0 / camera.viewport.x,
                           2.0 / camera.viewport.y);

    var offset_ndc = rad * in.corner * px2ndc;

    // let a = unpack2x16float(vertex.scale[0]);
    // offset_ndc.x *= a.x;
    // offset_ndc.y *= a.y;

    //offset_ndc *= 10.0; // IF SCENE HAS NO SCALE ATTRIBUTE

    let ndc_center = clipPos.xy / clipPos.w;
    let new_ndc = ndc_center + offset_ndc;
    let new_xyclip = new_ndc * clipPos.w;    

    out.position = vec4<f32>(new_xyclip, clipPos.z, clipPos.w);
    out.uv = in.corner * rad;
    out.color = splat.color;
    out.conic = splat.conic;
    out.opacity = splat.opacity;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {    
    let conX = in.conic.x;
    let conY = in.conic.y;
    let conZ = in.conic.z;
    let d = in.uv;

    
    let power = -0.5f * (conX * d.x * d.x + conZ * d.y * d.y) - conY * d.x * d.y;
    //let power = -0.5 * (conX * d.x * d.x + 2.0 * conY * d.x * d.y + conZ * d.y * d.y);
    if (power > 0.0) {
        discard;
    }
    let alpha = min(0.99, in.opacity * exp(power));
    if (alpha < 1.0 / 255.0) {
        discard;
    }

    return vec4<f32>(in.color * alpha, alpha);
}
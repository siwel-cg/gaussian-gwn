const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;

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

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    NDCpos: vec4<f32>,
    conic: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    opacity: f32
};

//TODO: bind your data here

// CAMERA AND POINT GUASSIAN DATA:
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> gaussians : array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_coeffs: array<u32>;
@group(1) @binding(2)
var<uniform> gauss_mult: f32;

// SORTS
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// SPLAT BINGS
@group(3) @binding(0)
var<storage, read_write> splatList : array<Splat>;

@group(3) @binding(1)
var<storage, read_write> splatIndexList : array<u32>;

@group(3) @binding(2)
var<storage, read_write> indirect_params: array<u32>;


/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base_u32_idx = splat_idx * 24u;
    let coef_offset = c_idx * 3u;
    
    let u32_offset = coef_offset / 2u;
    let is_even = (coef_offset % 2u) == 0u;
    
    let packed0 = sh_coeffs[base_u32_idx + u32_offset];
    let packed1 = sh_coeffs[base_u32_idx + u32_offset + 1u];
    
    if (is_even) {
        let rg = unpack2x16float(packed0);
        let bx = unpack2x16float(packed1);
        return vec3<f32>(rg.x, rg.y, bx.x);
    } else {
        let xr = unpack2x16float(packed0);
        let gb = unpack2x16float(packed1);
        return vec3<f32>(xr.y, gb.x, gb.y);
    }
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    //TODO: set up pipeline as described in instruction

    // IDENTITY OPS ON SORT BUFFERS SO THEY DON"T GET CUT
    if (idx == 0u) {
    //     let np = sort_infos.passes;
    //     sort_infos.passes = np;

    //     if (arrayLength(&sort_depths) > 0u) {
    //         let v = sort_depths[0];
    //         sort_depths[0] = v;
    //     }

    //     if (arrayLength(&sort_indices) > 0u) {
    //         let i = sort_indices[0];
    //         sort_indices[0] = i;
    //     }

        let dy = sort_dispatch.dispatch_y;
        sort_dispatch.dispatch_y = dy;
    }

    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // WE GONNA START BY JUST OUTPUTTING SPLAT DATA
    let vertex = gaussians[idx]; 
    let a = unpack2x16float(vertex.pos_opacity[0]);
    let b = unpack2x16float(vertex.pos_opacity[1]);
    let pos = vec4<f32>(a.x, a.y, b.x, 1.);

    let s1 = unpack2x16float(vertex.scale[0]);
    let s2 = unpack2x16float(vertex.scale[1]);
    //let scale = vec4<f32>(s1.x, s1.y, s2.x, 1.0);
    let scale = exp(vec3<f32>(s1.x, s1.y, s2.x));

    let r1 = unpack2x16float(vertex.rot[0]);
    let r2 = unpack2x16float(vertex.rot[1]);
    let rot = vec4<f32>(r1.x, r1.y, r2.x, r2.y);

    let viewPos = camera.view *  pos;
    let clipPos = camera.proj * viewPos;

    // CULLING
    if (clipPos.w  > 0.0 &&  
        clipPos.x >= -clipPos.w * 1.2 && clipPos.x <= clipPos.w * 1.2 &&
        clipPos.y >= -clipPos.w * 1.2 && clipPos.y <= clipPos.w * 1.2 && 
        clipPos.z >= 0.0 && clipPos.z <= clipPos.w ) {
        
        // CALCULATE COV
        let normRot = normalize(rot);
        let r = normRot.x;
        let x = normRot.y;
        let y = normRot.z;
        let z = normRot.w;

        // let R = mat3x3f(
        //     1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        //     2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        //     2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
        // );

        let R = transpose(mat3x3f(
            1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
            2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
            2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
        ));

        let gaussian_mult = gauss_mult; // CHANGE TO GAUSSIAN_MULTIPLIER LATER
        let S = mat3x3f(
            gaussian_mult * scale.x, 0.0, 0.0,
            0.0, gaussian_mult * scale.y, 0.0,
            0.0, 0.0, gaussian_mult * scale.z
        );

        // let M = R * S;
        // let Sigma3D = M * transpose(M);
        let M = R * S;


        let W = (mat3x3f(
            camera.view[0].xyz,
            camera.view[1].xyz,
            camera.view[2].xyz
        ));

        let M_cam = (W) * M;  // or W * M
        let Sigma3D_cam = M_cam * transpose(M_cam);
        
        // Use PIXEL focal lengths, not projection matrix values!
        let fx = camera.focal.x;  // focal_x stored in focal[1]
        let fy = -camera.focal.y;  // focal_y stored in focal[0]


        let tz = viewPos.z;
        let tz2 = tz * tz;

        let J = mat3x3f(
            fx / tz, 0.0, -fx * viewPos.x / tz2,
            0.0, fy / tz, -fy * viewPos.y / tz2,
            0.0, 0.0, 0.0
        );

        //let T = J * W;
        //var Sigma2D = T * Sigma3D * transpose(T);
        var Sigma2D = J * Sigma3D_cam * transpose(J);

        // Low-pass filter (in pixel space, 0.3 pixels is reasonable)
        // Sigma2D[0][0] += 0.3;
        // Sigma2D[1][1] += 0.3;

        let cov = vec3f(Sigma2D[0][0], Sigma2D[0][1], Sigma2D[1][1]);
        let det = (cov.x * cov.z - cov.y * cov.y);
        if (det == 0.0f) { return; }

        // CONIC
        let det_inv = 1.f / det;
        let conic = vec3f(cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv);

        // RADIUS
        let mid = 0.5f * (cov.x + cov.z);
        let lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
        let lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
        let radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

        // SH COLLORS
        let cam_pos = camera.view_inv[3].xyz;
        let view_dir = normalize(cam_pos - pos.xyz);
        let sh_deg = 3u; // HARD CODED DEG FOR NOW, USE IN COMBO WITH THE OTHER THING
        let color = computeColorFromSH(view_dir, idx, sh_deg);

        // CHECK FOR BAD VALUES
        if (det <= 0.0001f) { 
            return;
        }

        let culledIdx = atomicAdd(&sort_infos.keys_size, 1u);
        if (culledIdx >= arrayLength(&splatIndexList)) {
            return;
        }

        splatIndexList[culledIdx] = idx;
        splatList[idx].NDCpos = clipPos / clipPos.w;
        splatList[idx].conic = conic;
        splatList[idx].radius = radius;
        splatList[idx].color = color;
        //splatList[idx].opacity = b.y;
        splatList[idx].opacity = 1.0 / (1.0 + exp(-b.y));

        // STORE DEPTH AND INDICES FOR SORTING LATTER
        let u = bitcast<u32>(-viewPos.z);
        let mask = select(0xFFFFFFFFu, 0x80000000u, (u & 0x80000000u) == 0u);
        let depth_uint = u ^ mask;
        sort_depths[culledIdx] = depth_uint;
        sort_indices[culledIdx] = culledIdx;
    }

    if (idx == 0u) {
        indirect_params[0] = 6u;
        indirect_params[1] = atomicLoad(&sort_infos.keys_size);  
        indirect_params[2] = 0u;
        indirect_params[3] = 0u;
    }

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
}
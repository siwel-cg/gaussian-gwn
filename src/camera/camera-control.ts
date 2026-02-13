// camera-control.ts
import { vec3, Vec3, mat4 } from 'wgpu-matrix';
import { Camera } from './camera';

export class CameraControl {
  element: HTMLCanvasElement;
  
  private yaw: number = 0;
  private pitch: number = 0;
  private distance: number = 0;
  
  target: Vec3 = vec3.create(0, 0, 0);
  
  // Store initial values for reset
  private initialTarget: Vec3 = vec3.create(0, 0, 0);
  private initialDistance: number = 5.0;
  
  constructor(private camera: Camera) {
    this.register_element(camera.canvas);
    this.registerKeyboard();
    this.updateCamera();
  }

  register_element(value: HTMLCanvasElement) {
    if (this.element && this.element != value) {
      this.element.removeEventListener('pointerdown', this.downCallback);
      this.element.removeEventListener('pointermove', this.moveCallback);
      this.element.removeEventListener('pointerup', this.upCallback);
      this.element.removeEventListener('wheel', this.wheelCallback);
    }

    this.element = value;
    
    // Use arrow functions to preserve 'this'
    this.downCallback = this.downCallback.bind(this);
    this.moveCallback = this.moveCallback.bind(this);
    this.upCallback = this.upCallback.bind(this);
    this.wheelCallback = this.wheelCallback.bind(this);
    
    this.element.addEventListener('pointerdown', this.downCallback);
    this.element.addEventListener('pointermove', this.moveCallback);
    this.element.addEventListener('pointerup', this.upCallback);
    this.element.addEventListener('wheel', this.wheelCallback);
    this.element.addEventListener('contextmenu', (e) => { e.preventDefault(); });
  }

  registerKeyboard() {
    document.addEventListener('keydown', (e) => {
      if (e.key === 'f' || e.key === 'F') {
        this.resetToInitial();
      }
      if (e.key === 'ArrowLeft') {
        this.yaw += 0.05;
        this.updateCamera();
      }
      if (e.key === 'ArrowRight') {
        this.yaw -= 0.05;
        this.updateCamera();
      }
    });
  }

  private panning = false;
  private rotating = false;
  private lastX: number;
  private lastY: number;

  downCallback(event: PointerEvent) {
    if (!event.isPrimary) return;

    if (event.button === 0) {
      this.rotating = true;
      this.panning = false;
    } else {
      this.rotating = false;
      this.panning = true;
    }
    this.lastX = event.pageX;
    this.lastY = event.pageY;
  }

  moveCallback(event: PointerEvent) {
    if (!(this.rotating || this.panning)) return;

    const xDelta = event.pageX - this.lastX;
    const yDelta = event.pageY - this.lastY;
    this.lastX = event.pageX;
    this.lastY = event.pageY;

    if (this.rotating) {
      this.rotate(xDelta, yDelta);
    } else if (this.panning) {
      this.pan(xDelta, yDelta);
    }
  }

  upCallback(event: PointerEvent) {
    this.rotating = false;
    this.panning = false;
    event.preventDefault();
  }

  wheelCallback(event: WheelEvent) {
    event.preventDefault();
    this.distance += event.deltaY * 0.01;
    this.distance = Math.max(0.1, this.distance);
    this.updateCamera();
  }

  setTarget(target: Vec3, distance?: number) {
    vec3.copy(target, this.target);
    vec3.copy(target, this.initialTarget);
    
    if (distance !== undefined) {
      this.distance = distance;
      this.initialDistance = distance;
    }
    this.yaw = 0;
    this.pitch = 0;
    this.updateCamera();
  }

  resetToInitial() {
    vec3.copy(this.initialTarget, this.target);
    this.distance = this.initialDistance;
    this.yaw = 0;
    this.pitch = 0;
    this.updateCamera();
    console.log('Camera reset to initial position');
  }

  private updateCamera() {
    const maxPitch = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));

    // Spherical coordinates around target
    const x = this.distance * Math.cos(this.pitch) * Math.sin(this.yaw);
    const y = this.distance * Math.sin(this.pitch);
    const z = this.distance * Math.cos(this.pitch) * Math.cos(this.yaw);

    this.camera.position[0] = this.target[0] + x;
    this.camera.position[1] = this.target[1] + y;
    this.camera.position[2] = this.target[2] + z;

    // Forward: from camera to target (normalized)
    const forward = vec3.normalize(vec3.sub(this.target, this.camera.position));
    
    const worldUp = vec3.create(0, 1, 0);
    const right = vec3.normalize(vec3.cross(forward, worldUp));
    const up = vec3.cross(right, forward);

    // Flip: camera looks along +Z instead of -Z
    this.camera.rotation[0] = right[0];
    this.camera.rotation[1] = up[0];
    this.camera.rotation[2] = forward[0];
    this.camera.rotation[3] = 0;

    this.camera.rotation[4] = right[1];
    this.camera.rotation[5] = up[1];
    this.camera.rotation[6] = forward[1];
    this.camera.rotation[7] = 0;

    this.camera.rotation[8] = right[2];
    this.camera.rotation[9] = up[2];
    this.camera.rotation[10] = forward[2];
    this.camera.rotation[11] = 0;

    this.camera.rotation[12] = 0;
    this.camera.rotation[13] = 0;
    this.camera.rotation[14] = 0;
    this.camera.rotation[15] = 1;

    this.camera.update_buffer();
  }

  rotate(xDelta: number, yDelta: number) {
    const sensitivity = 0.005;
    this.yaw -= xDelta * sensitivity;
    this.pitch += yDelta * sensitivity;
    console.log(`rotate called: xDelta=${xDelta}, yDelta=${yDelta}`);
    this.updateCamera();
  }

  pan(xDelta: number, yDelta: number) {
    const panSpeed = this.distance * 0.002;
    
    const right = vec3.create(
      this.camera.rotation[0],
      this.camera.rotation[1],
      this.camera.rotation[2]
    );
    const up = vec3.create(
      this.camera.rotation[4],
      this.camera.rotation[5],
      this.camera.rotation[6]
    );

    vec3.addScaled(this.target, right, -xDelta * panSpeed, this.target);
    vec3.addScaled(this.target, up, -yDelta * panSpeed, this.target);
    
    this.updateCamera();
  }

}
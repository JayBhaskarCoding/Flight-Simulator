//! Windowed mode: winit 0.30 event loop + wgpu surface presentation.
//!
//! The windowed path shares [`crate::sim::SimCore`] with headless mode
//! — the physics thread is identical; only presentation differs.

use crate::ecs::Ecs;
use crate::sim::{
    build_frame_input, camera_ecef_pose, spawn_aircraft_and_camera, system_build_draws,
    system_camera_follow, system_sync_aircraft, ControlCommand, SimCore, Snapshot,
};
use glam::DVec3;
use sim_render::Renderer;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Preferred present formats (must be sRGB — the tonemap shader emits
/// linear ACES and relies on the target's transfer function).
const TARGET_FORMAT_PREFERENCE: [wgpu::TextureFormat; 2] = [
    wgpu::TextureFormat::Bgra8UnormSrgb,
    wgpu::TextureFormat::Rgba8UnormSrgb,
];

/// Shared inputs written from the winit thread.
pub struct WindowInput {
    pub command: Arc<Mutex<ControlCommand>>,
    pub snapshot: Arc<RwLock<Snapshot>>,
}

struct FrameResources {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    window: Arc<Window>,
}

pub struct App {
    input: WindowInput,
    bridge: crate::sim::FrameBridge,
    sky_sun_enu: DVec3,
    model: Option<sim_render::GltfModel>,
    ecs: Ecs,
    resources: Option<FrameResources>,
    start: Instant,
    pending_resize: Option<(u32, u32)>,
}

impl App {
    /// Build the windowed application, seeding the ECS with one
    /// aircraft entity (driven by the physics snapshot) and one chase
    /// camera.
    pub fn new(
        input: WindowInput,
        core: &SimCore,
        sky_sun_enu: DVec3,
        model: Option<sim_render::GltfModel>,
        mut ecs: Ecs,
    ) -> Self {
        spawn_aircraft_and_camera(&mut ecs);

        Self {
            input,
            bridge: core.bridge,
            sky_sun_enu,
            model,
            ecs,
            resources: None,
            start: Instant::now(),
            pending_resize: None,
        }
    }

    /// Run the windowed event loop (blocks until the window closes).
    pub fn run(mut self) -> Result<(), String> {
        let event_loop = EventLoop::new().map_err(|e| format!("event loop: {e}"))?;
        event_loop.run_app(&mut self).map_err(|e| format!("run_app: {e}"))?;
        Ok(())
    }

    fn render_frame(&mut self) -> Result<(), String> {
        let snap = self
            .input
            .snapshot
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();

        // ECS systems for this frame
        system_sync_aircraft(&mut self.ecs, &snap);
        system_camera_follow(&mut self.ecs);

        let cam_entity = self.ecs.camera_entities().first().copied();
        let (cam_pos_enu, cam_quat_enu) = cam_entity
            .and_then(|c| self.ecs.transform.get(c).map(|t| (t.position, t.orientation)))
            .unwrap_or((snap.position_enu, snap.orientation_enu));
        let (camera_ecef, camera_quat) = camera_ecef_pose(&self.bridge, cam_pos_enu, cam_quat_enu);

        let draws =
            system_build_draws(&self.ecs, &self.bridge, camera_ecef, camera_quat, self.model.as_ref());

        let time_s = self.start.elapsed().as_secs_f64();
        let mut frame =
            build_frame_input(&self.bridge, camera_ecef, camera_quat, &snap, self.sky_sun_enu, time_s);
        frame.draws = &draws;

        let res = self.resources.as_ref().ok_or("frame resources not initialised")?;
        let surface_tex = match res.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                res.surface.configure(&res.renderer.device_handle(), &res.surface_config);
                return Ok(());
            }
            Err(e) => return Err(format!("surface: {e}")),
        };
        let view =
            surface_tex.texture.create_view(&wgpu::TextureViewDescriptor::default());
        res.renderer
            .render(&view, res.surface_config.format, &frame)
            .map_err(|e| format!("render: {e}"))?;
        res.window.pre_present_notify();
        surface_tex.present();
        Ok(())
    }

    fn apply_resize(&mut self, width: u32, height: u32) {
        let Some(res) = self.resources.as_mut() else {
            self.pending_resize = Some((width, height));
            return;
        };
        res.surface_config.width = width.max(1);
        res.surface_config.height = height.max(1);
        res.surface.configure(&res.renderer.device_handle(), &res.surface_config);
        res.renderer.resize(width, height);
    }

    fn handle_key(&self, key: &Key, pressed: bool) -> Option<ControlCommand> {
        // Returns the *new* command state for a handled key: held keys
        // deflect, released keys spring back to neutral.
        let mut cmd = *self.input.command.lock().ok()?;
        let max = crate::sim::MAX_ELEVATOR_RAD;
        let ail = crate::sim::MAX_AILERON_RAD;
        let rud = crate::sim::MAX_RUDDER_RAD;
        let ch = match key {
            Key::Character(c) => c.as_str(),
            _ => return None,
        };
        match ch {
            "w" | "W" => cmd.elevator_rad = if pressed { -max } else { 0.0 },
            "s" | "S" => cmd.elevator_rad = if pressed { max } else { 0.0 },
            "a" | "A" => cmd.aileron_rad = if pressed { -ail } else { 0.0 },
            "d" | "D" => cmd.aileron_rad = if pressed { ail } else { 0.0 },
            "q" | "Q" => cmd.rudder_rad = if pressed { -rud } else { 0.0 },
            "e" | "E" => cmd.rudder_rad = if pressed { rud } else { 0.0 },
            "b" | "B" => cmd.brake = if pressed { 1.0 } else { 0.0 },
            "g" | "G" if pressed => cmd.flap_step = (cmd.flap_step + 1).min(3),
            _ => return None,
        }
        Some(cmd)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.resources.is_some() {
            return;
        }
        let result = block_on_future(self.init_surface(event_loop));
        if let Err(e) = result {
            eprintln!("window init failed: {e}");
            event_loop.exit();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.apply_resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    event_loop.exit();
                    return;
                }
                if let Some(cmd) = self.handle_key(&event.logical_key, event.state == ElementState::Pressed)
                {
                    if let Ok(mut guard) = self.input.command.lock() {
                        *guard = cmd;
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some((w, h)) = self.pending_resize.take() {
                    self.apply_resize(w, h);
                }
                if let Err(e) = self.render_frame() {
                    eprintln!("render error: {e}");
                }
                if let Some(res) = self.resources.as_ref() {
                    res.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

impl App {
    /// Async surface/adapter/device setup (driven by the small
    /// park-waker executor from `resumed`).
    async fn init_surface(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let window: Arc<Window> = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("OpenMSFS — clean-room flight simulator")
                        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0)),
                )
                .map_err(|e| format!("create_window: {e}"))?,
        );

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("create_surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or("no wgpu adapter for the window surface")?;
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("open-msfs-window"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = TARGET_FORMAT_PREFERENCE
            .into_iter()
            .find(|f| caps.formats.contains(f))
            .ok_or_else(|| format!("surface offers no sRGB format (got {:?})", caps.formats))?;
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let renderer =
            Renderer::with_device(device, queue, surface_config.width, surface_config.height)
                .map_err(|e| format!("renderer: {e}"))?;
        window.request_redraw();
        self.resources = Some(FrameResources { surface, surface_config, renderer, window });
        Ok(())
    }
}

/// Minimal future executor for `resumed` (the winit 0.30 callback API
/// is synchronous while surface/adapter acquisition is async).
fn block_on_future<F: std::future::Future>(fut: F) -> F::Output {
    use std::sync::Arc as StdArc;
    use std::task::{Context, RawWaker, RawWakerVTable, Waker};
    struct Park(std::thread::Thread);
    const VT: RawWakerVTable = RawWakerVTable::new(
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            let c = StdArc::into_raw(arc.clone()) as *const ();
            let _ = StdArc::into_raw(arc);
            RawWaker::new(c, &VT)
        },
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            arc.0.unpark();
        },
        |ptr| {
            let arc = unsafe { StdArc::from_raw(ptr as *const Park) };
            arc.0.unpark();
            let _ = StdArc::into_raw(arc);
        },
        |ptr| drop(unsafe { StdArc::from_raw(ptr as *const Park) }),
    );
    let mut fut = Box::pin(fut);
    let waker = {
        let arc = StdArc::new(Park(std::thread::current()));
        unsafe { Waker::from_raw(RawWaker::new(StdArc::into_raw(arc) as *const (), &VT)) }
    };
    let mut cx = Context::from_waker(&waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => {
                std::thread::park_timeout(std::time::Duration::from_millis(5))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::SimConfig;

    #[test]
    fn key_press_deflects_and_release_centers() {
        let input = WindowInput {
            command: Arc::new(Mutex::new(ControlCommand::default())),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let cfg = SimConfig::default();
        let core = SimCore::new(cfg).expect("core");
        let app = App::new(input, &core, DVec3::new(0.0, 0.5, 0.8), None, Ecs::new());

        let cmd = app
            .handle_key(&Key::Character("w".into()), true)
            .expect("handled");
        assert!((cmd.elevator_rad + crate::sim::MAX_ELEVATOR_RAD).abs() < 1e-9);
        let cmd = app.handle_key(&Key::Character("w".into()), false).expect("handled");
        assert_eq!(cmd.elevator_rad, 0.0);
        assert!(app.handle_key(&Key::Character("z".into()), true).is_none());
    }

    #[test]
    fn seeds_ecs_with_aircraft_and_chase_camera() {
        let input = WindowInput {
            command: Arc::new(Mutex::new(ControlCommand::default())),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
        };
        let core = SimCore::new(SimConfig::default()).expect("core");
        let app = App::new(input, &core, DVec3::new(0.0, 0.5, 0.8), None, Ecs::new());
        assert_eq!(app.ecs.aircraft_entities().len(), 1);
        assert_eq!(app.ecs.camera_entities().len(), 1);
        assert_eq!(app.ecs.renderable_entities().len(), 1);
    }
}

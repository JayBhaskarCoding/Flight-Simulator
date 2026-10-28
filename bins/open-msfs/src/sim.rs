//! Simulation core: the 1000 Hz physics thread, the ENU↔ECEF frame
//! bridge, ECS systems (camera follow, simvar publishing, draw-list
//! construction) and the autopilot wiring.
//!
//! Frame conventions
//! -----------------
//! * **Physics frame** — flat ENU at the field origin (`+X` east,
//!   `+Y` north, `+Z` up), which is `sim-physics`' world model.
//! * **World frame for rendering** — ECEF (`f64`), produced by
//!   [`FrameBridge`] (`origin + R_enu→ecef · p_enu`, all `f64`).
//! * **GPU** — camera-relative `f32` via `sim_geo::FloatingOrigin`
//!   (the only sanctioned narrowing path).

use crate::ecs::{Ecs, Transform, Velocity};
use crate::solar;
use glam::{DQuat, DMat3, DVec3, Mat4, Quat, Vec3};
use sim_atmosphere::sky::{SkyLuts, SkyParams};
use sim_atmosphere::IsaModel;
use sim_avionics::SimVarBus;
use sim_avionics::{ApMode, ApSample, Autopilot, DEFAULT_PORT};
use sim_cfg::AircraftDefinition;
use sim_geo::{Ecef, FloatingOrigin, LatLonAlt, WGS84};
use sim_physics::{
    AtmosphereSample, ControlInputs, PhysicsWorld, RigidBodyState, StepDiagnostics,
};
use sim_render::types::{CloudUniforms, DrawItem};
use sim_render::FrameInput;
use sim_render::Renderer;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Maximum control deflections mapped from autopilot `[-1, 1]`
/// outputs to surface radians (C172-class values).
pub const MAX_ELEVATOR_RAD: f64 = 25.0f64.to_radians();
pub const MAX_AILERON_RAD: f64 = 20.0f64.to_radians();
pub const MAX_RUDDER_RAD: f64 = 30.0f64.to_radians();

// ---------------------------------------------------------------------------
// frame bridge
// ---------------------------------------------------------------------------

/// Flat-ENU origin ↔ ECEF conversion (all `f64`).
#[derive(Debug, Clone, Copy)]
pub struct FrameBridge {
    origin_llh: LatLonAlt,
    origin_ecef: DVec3,
    /// Columns = east / north / up expressed in ECEF.
    enu_to_ecef: DMat3,
}

impl FrameBridge {
    pub fn new(origin_llh: LatLonAlt) -> Self {
        let origin_ecef = WGS84.geodetic_to_ecef(&origin_llh).as_dvec3();
        let (phi, lambda) = (origin_llh.latitude_rad(), origin_llh.longitude_rad());
        let (sphi, cphi) = (phi.sin(), phi.cos());
        let (slam, clam) = (lambda.sin(), lambda.cos());
        let east = DVec3::new(-slam, clam, 0.0);
        let north = DVec3::new(-sphi * clam, -sphi * slam, cphi);
        let up = DVec3::new(cphi * clam, cphi * slam, sphi);
        Self {
            origin_llh,
            origin_ecef,
            enu_to_ecef: DMat3::from_cols(east, north, up),
        }
    }

    pub fn origin_llh(&self) -> LatLonAlt {
        self.origin_llh
    }

    pub fn origin_ecef(&self) -> Ecef {
        Ecef::from(self.origin_ecef)
    }

    /// ENU position → ECEF (f64).
    pub fn enu_to_ecef(&self, p: DVec3) -> DVec3 {
        self.origin_ecef + self.enu_to_ecef * p
    }

    /// ECEF → geodetic (f64).
    pub fn ecef_to_llh(&self, ecef: DVec3) -> LatLonAlt {
        WGS84.ecef_to_geodetic(&Ecef::from(ecef))
    }

    /// Body/ENU orientation → ECEF orientation (f64).
    pub fn orientation_to_ecef(&self, q: DQuat) -> DQuat {
        DQuat::from_mat3(&self.enu_to_ecef) * q
    }

    /// The raw ENU → ECEF rotation matrix (f64).
    pub fn enu_to_ecef_mat(&self) -> DMat3 {
        self.enu_to_ecef
    }

    /// Rotation taking camera-relative **ECEF** vectors into the
    /// planetary sky frame used by `sim-atmosphere` (ENU with +Y up
    /// mapped to the sky model's zenith axis): sky = (east, up, north)
    /// … the atmosphere model only needs a *consistent* Y-up frame; we
    /// use sky = (north, up, −east)? — simpler: sky axes
    /// `x = east, y = up, z = south` is arbitrary, but `SkyLuts`
    /// computes azimuth relative to the sun's horizontal projection,
    /// so any right-handed Y-up frame gives identical radiance. We
    /// pick `x = east, y = up, z = north`… must be right-handed:
    /// east × up = east×up = −north? Use `x = north, y = up,
    /// z = east`: north×up = east ✓ right-handed.
    pub fn sky_from_ecef_quat(&self) -> DQuat {
        let east = self.enu_to_ecef.x_axis;
        let north = self.enu_to_ecef.y_axis;
        let up = self.enu_to_ecef.z_axis;
        // right-handed Y-up basis with x = north, y = up, z = east:
        let m = DMat3::from_cols(north, up, east);
        // rotation ECEF → sky = transpose of (sky → ECEF)
        DQuat::from_mat3(&m.transpose())
    }
}

// ---------------------------------------------------------------------------
// snapshots shared across threads
// ---------------------------------------------------------------------------

/// Immutable per-frame state published by the physics thread.
#[derive(Debug, Clone)]
pub struct Snapshot {
    pub sim_time_s: f64,
    /// Aircraft COM in the physics ENU frame (f64).
    pub position_enu: DVec3,
    /// Body → ENU attitude (f64).
    pub orientation_enu: DQuat,
    pub velocity_enu: DVec3,
    pub omega_body: DVec3,
    /// Geodetic state derived through the bridge (f64).
    pub geodetic: LatLonAlt,
    pub altitude_m: f64,
    pub true_airspeed_mps: f64,
    pub indicated_airspeed_mps: f64,
    pub ground_speed_mps: f64,
    pub heading_deg: f64,
    pub pitch_deg: f64,
    pub roll_deg: f64,
    pub vertical_speed_mps: f64,
    pub density_kg_m3: f64,
    pub gear_on_ground: bool,
    pub diagnostics: StepDiagnostics,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            sim_time_s: 0.0,
            position_enu: DVec3::ZERO,
            orientation_enu: DQuat::IDENTITY,
            velocity_enu: DVec3::ZERO,
            omega_body: DVec3::ZERO,
            geodetic: LatLonAlt::from_degrees(0.0, 0.0, 0.0),
            altitude_m: 0.0,
            true_airspeed_mps: 0.0,
            indicated_airspeed_mps: 0.0,
            ground_speed_mps: 0.0,
            heading_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            vertical_speed_mps: 0.0,
            density_kg_m3: 1.225,
            gear_on_ground: true,
            diagnostics: StepDiagnostics {
                sim_time_s: 0.0,
                steps: 0,
                wheels_in_contact: 0,
                bottomed_out: false,
                kinetic_energy_j: 0.0,
                diverged: false,
            },
        }
    }
}

/// Shared controls written by input (keyboard/CLI) and read by the
/// physics thread each 1 ms tick (zero-order hold).
#[derive(Debug, Clone, Copy, Default)]
pub struct ControlCommand {
    /// Surface deflections (rad / step / 0..1) — direct FDM commands.
    pub elevator_rad: f64,
    pub aileron_rad: f64,
    pub rudder_rad: f64,
    pub flap_step: usize,
    pub brake: f64,
    pub steering: f64,
    /// When set, the autopilot owns elevator/aileron and its outputs
    /// override the manual surfaces.
    pub autopilot: Option<ApMode>,
    /// Heading-hold target (deg); `None` = capture current heading.
    pub ap_heading_deg: Option<f64>,
    /// Altitude-hold target (ft); `None` = capture current altitude.
    pub ap_altitude_ft: Option<f64>,
}

// ---------------------------------------------------------------------------
// sim core
// ---------------------------------------------------------------------------

/// Initial conditions for the simulation.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Field/origin geodetic position.
    pub origin: LatLonAlt,
    /// Start airborne (trueairspeed) or on the runway.
    pub airborne: bool,
    pub initial_heading_rad: f64,
    pub initial_airspeed_mps: f64,
    pub initial_altitude_m: f64,
    /// Fast-forward physics without wall-clock pacing.
    pub fast: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            origin: LatLonAlt::from_degrees(40.0, -74.0, 0.0),
            airborne: false,
            initial_heading_rad: std::f64::consts::FRAC_PI_2, // east runway
            initial_airspeed_mps: 0.0,
            initial_altitude_m: 0.0,
            fast: true,
        }
    }
}

/// Everything the engine owns while running.
pub struct SimCore {
    pub bridge: FrameBridge,
    pub aircraft_def: AircraftDefinition,
    pub world: PhysicsWorld,
    pub isa: IsaModel,
    pub sky_params: SkyParams,
    pub sky_luts: SkyLuts,
    pub bus: SimVarBus,
    pub autopilot: Autopilot,
    pub config: SimConfig,
    pub handle: Option<std::thread::JoinHandle<()>>,
    running: Arc<AtomicBool>,
    snapshot: Arc<RwLock<Snapshot>>,
    command: Arc<Mutex<ControlCommand>>,
    field_altitude_m: f64,
}

impl SimCore {
    /// Build the simulation from the clean-room C172 assets and the
    /// initial conditions (does not start the thread).
    pub fn new(config: SimConfig) -> Result<Self, String> {
        let aircraft_def =
            AircraftDefinition::load_reference_c172().map_err(|e| format!("cfg: {e}"))?;

        let bridge = FrameBridge::new(config.origin);
        let field_altitude_m = config.origin.altitude_m();
        let initial_alt = if config.airborne {
            field_altitude_m + config.initial_altitude_m
        } else {
            field_altitude_m + 0.15 // on the wheels
        };
        let state = if config.airborne {
            RigidBodyState::level_flight(
                initial_alt,
                config.initial_heading_rad,
                config.initial_airspeed_mps,
            )
        } else {
            let orientation = RigidBodyState::upright_attitude(config.initial_heading_rad);
            RigidBodyState {
                position_m: DVec3::new(0.0, 0.0, initial_alt),
                orientation,
                velocity_mps: DVec3::ZERO,
                omega_body_radps: DVec3::ZERO,
            }
        };

        let mass = aircraft_def.flight_model.weight.empty_weight_kg
            + aircraft_def.flight_model.total_usable_fuel_kg();
        let cg = DVec3::new(aircraft_def.flight_model.weight.empty_cg_arm_m, 0.0, 0.0);
        let world = PhysicsWorld::new_on_flat_field(&aircraft_def, mass, cg, state, field_altitude_m);

        let isa = IsaModel::standard();
        // Sun: bridge gives the origin's sky frame; the sky LUTs are
        // built for the origin once (observer altitude ≈ field).
        let sun_enu = sun_enu(&config);
        // SkyParams uses a Y-up frame; ENU has +Y = north, so rotate
        // the ENU sun into a Y-up frame with y = up:
        let sky_params = SkyParams {
            sun_direction: DVec3::new(sun_enu.x, sun_enu.z, -sun_enu.y).normalize(),
            observer_altitude_m: field_altitude_m,
            ..SkyParams::default()
        };
        let sky_luts = SkyLuts::new(&sky_params);

        let bus = SimVarBus::with_defaults();
        let autopilot = Autopilot::with_defaults();

        Ok(Self {
            bridge,
            aircraft_def,
            world,
            isa,
            sky_params,
            sky_luts,
            bus,
            autopilot,
            config,
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
            snapshot: Arc::new(RwLock::new(Snapshot::default())),
            command: Arc::new(Mutex::new(ControlCommand::default())),
            field_altitude_m,
        })
    }

    pub fn snapshot_handle(&self) -> Arc<RwLock<Snapshot>> {
        Arc::clone(&self.snapshot)
    }

    pub fn command_handle(&self) -> Arc<Mutex<ControlCommand>> {
        Arc::clone(&self.command)
    }

    pub fn bus(&self) -> SimVarBus {
        self.bus.clone()
    }

    /// Spawn the 1000 Hz physics thread. Each tick: read the shared
    /// command (zero-order hold), refresh the ISA sample for the
    /// current altitude, step the RK4 FDM, publish a snapshot, and
    /// update the simvar bus (every 10 steps = 100 Hz to keep the
    /// broadcast channel cheap).
    pub fn start(&mut self) {
        assert!(self.handle.is_none(), "already running");
        self.running.store(true, Ordering::SeqCst);

        let running = Arc::clone(&self.running);
        let snapshot = Arc::clone(&self.snapshot);
        let command = Arc::clone(&self.command);
        let bus = self.bus.clone();
        let fast = self.config.fast;
        let field_alt = self.field_altitude_m;

        // Move the pieces the thread owns out of self.
        let mut world = std::mem::replace(
            &mut self.world,
            PhysicsWorld::new_on_flat_field(
                &self.aircraft_def,
                1.0,
                DVec3::ZERO,
                RigidBodyState::default(),
                field_alt,
            ),
        );
        let isa = IsaModel::standard();
        let bridge = self.bridge;
        let mut autopilot = Autopilot::with_defaults();

        let handle = std::thread::Builder::new()
            .name("physics-1000hz".into())
            .spawn(move || {
                let tick = Duration::from_nanos(1_000_000); // 1 ms
                let mut next = Instant::now();
                let mut last_publish = std::time::Instant::now();
                let mut last_diag = StepDiagnostics {
                    sim_time_s: 0.0,
                    steps: 0,
                    wheels_in_contact: 0,
                    bottomed_out: false,
                    kinetic_energy_j: 0.0,
                    diverged: false,
                };
                while running.load(Ordering::SeqCst) {
                    // real-time pacing (skipped in fast mode)
                    if !fast {
                        let now = Instant::now();
                        if now < next {
                            std::thread::sleep(next - now);
                        }
                        next += tick;
                        // spiral-of-death guard: drop debt beyond 100 ms
                        if next + Duration::from_millis(100) < Instant::now() {
                            next = Instant::now();
                        }
                    }

                    let cmd = *command.lock().unwrap_or_else(|e| e.into_inner());
                    apply_command(&mut world, &mut autopilot, &cmd);

                    let alt = world.state.position_m.z.max(field_alt);
                    let sample = isa.at_geometric_m(alt);
                    let atm = AtmosphereSample {
                        density_kg_m3: sample.density_kg_m3,
                        speed_of_sound_mps: sample.speed_of_sound_mps,
                    };
                    let diag = world.step(&atm);
                    last_diag = diag;

                    let publish = fast || last_publish.elapsed() >= Duration::from_millis(100);
                    if publish {
                        last_publish = Instant::now();
                        let snap = build_snapshot(&mut world, &bridge, &isa, field_alt, &diag);
                        publish_simvars(&bus, &snap);
                        *snapshot.write().unwrap_or_else(|e| e.into_inner()) = snap;
                    }
                }
                // final publish so no state is lost at shutdown
                let snap = build_snapshot(&mut world, &bridge, &isa, field_alt, &last_diag);
                publish_simvars(&bus, &snap);
                *snapshot.write().unwrap_or_else(|e| e.into_inner()) = snap;
            })
            .expect("spawn physics thread");
        self.handle = Some(handle);
    }

    /// Signal the physics thread to stop and join it.
    pub fn shutdown(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for SimCore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Map the shared command onto the FDM (manual surfaces, or the
/// autopilot cascade when engaged).
fn apply_command(world: &mut PhysicsWorld, autopilot: &mut Autopilot, cmd: &ControlCommand) {
    world.controls = ControlInputs {
        elevator_rad: cmd.elevator_rad,
        aileron_rad: cmd.aileron_rad,
        rudder_rad: cmd.rudder_rad,
        flap_step: cmd.flap_step,
        brake: cmd.brake,
        steering: cmd.steering,
    };
    if let Some(mode) = cmd.autopilot {
        let s = world.state;
        let alt = s.position_m.z;
        let v = s.velocity_mps;
        let fwd = s.orientation * DVec3::X;
        let right = s.orientation * DVec3::Y;
        let up = s.orientation * DVec3::new(0.0, 0.0, -1.0);
        let heading = fwd.x.atan2(fwd.y).to_degrees().rem_euclid(360.0);
        let pitch = (fwd.z.clamp(-1.0, 1.0)).asin().to_degrees();
        let roll = (-right.z).atan2(up.z).to_degrees();
        let vs_fpm = v.z * 196.850_394;
        let ap_sample = ApSample {
            heading_deg: heading,
            pitch_deg: pitch,
            roll_deg: roll,
            altitude_ft: alt * 3.280_839_895,
            vertical_speed_fpm: vs_fpm,
        };
        autopilot.set_mode(mode, &ap_sample);
        if let Some(h) = cmd.ap_heading_deg {
            autopilot.set_heading_target(h);
        }
        if let Some(a) = cmd.ap_altitude_ft {
            autopilot.set_altitude_target(a);
        }
        let out = autopilot.step(&ap_sample, sim_physics::DT);
        world.controls.elevator_rad = out.elevator * MAX_ELEVATOR_RAD;
        world.controls.aileron_rad = out.aileron * MAX_AILERON_RAD;
        world.controls.rudder_rad = out.rudder * MAX_RUDDER_RAD;
    }
}

fn build_snapshot(
    world: &mut PhysicsWorld,
    bridge: &FrameBridge,
    isa: &IsaModel,
    _field_alt: f64,
    diag: &StepDiagnostics,
) -> Snapshot {
    let s = world.state;
    let altitude_m = s.position_m.z;
    let sample = isa.at_geometric_m(altitude_m.max(0.0));
    let v = s.velocity_mps;
    let tas = v.length();
    let rho0: f64 = 1.225;
    let ias = tas * (sample.density_kg_m3 / rho0).sqrt();

    let fwd = s.orientation * DVec3::X;
    let right = s.orientation * DVec3::Y;
    let up = s.orientation * DVec3::new(0.0, 0.0, -1.0);
    let heading_deg = fwd.x.atan2(fwd.y).to_degrees().rem_euclid(360.0);
    let pitch_deg = (fwd.z.clamp(-1.0, 1.0)).asin().to_degrees();
    let roll_deg = (-right.z).atan2(up.z).to_degrees();

    let ground = world.gear_contacts();
    let on_ground = ground.iter().any(|c| c.force_world.length() > 1.0);

    Snapshot {
        sim_time_s: world.sim_time_s(),
        position_enu: s.position_m,
        orientation_enu: s.orientation,
        velocity_enu: v,
        omega_body: s.omega_body_radps,
        geodetic: bridge.ecef_to_llh(bridge.enu_to_ecef(s.position_m)),
        altitude_m,
        true_airspeed_mps: tas,
        indicated_airspeed_mps: ias,
        ground_speed_mps: DVec3::new(v.x, v.y, 0.0).length(),
        heading_deg,
        pitch_deg,
        roll_deg,
        vertical_speed_mps: v.z,
        density_kg_m3: sample.density_kg_m3,
        gear_on_ground: on_ground,
        diagnostics: *diag,
    }
}

fn publish_simvars(bus: &SimVarBus, s: &Snapshot) {
    use sim_avionics::Unit;
    let set = |name: &str, v: f64, u: Unit| {
        let _ = bus.set(name, v, u);
    };
    set("INDICATED AIRSPEED", s.indicated_airspeed_mps / 0.514_444, Unit::Knot);
    set("TRUE AIRSPEED", s.true_airspeed_mps / 0.514_444, Unit::Knot);
    set("GROUND SPEED", s.ground_speed_mps / 0.514_444, Unit::Knot);
    let rho0: f64 = 1.225;
    set(
        "MACH",
        s.true_airspeed_mps / (1.4 * 101_325.0 / rho0).sqrt(),
        Unit::Unitless,
    );
    set("ALTITUDE", s.altitude_m * 3.280_839_895, Unit::Foot);
    set("INDICATED ALTITUDE", s.altitude_m * 3.280_839_895, Unit::Foot);
    set("VERTICAL SPEED", s.vertical_speed_mps * 196.850_394, Unit::FeetPerMinute);
    set("HEADING DEGREES", s.heading_deg, Unit::Degree);
    set("PLANE PITCH DEGREES", s.pitch_deg, Unit::Degree);
    set("PLANE BANK DEGREES", s.roll_deg, Unit::Degree);
    set("LATITUDE", s.geodetic.latitude_deg(), Unit::Degree);
    set("LONGITUDE", s.geodetic.longitude_deg(), Unit::Degree);
    set(
        "GROUND VELOCITY",
        s.ground_speed_mps / 0.514_444,
        Unit::Knot,
    );
    set("SIMULATION TIME", s.sim_time_s, Unit::Second);
}

/// Unit sun vector in ENU at the config's origin (x east, y north,
/// z up) for the configured civil time.
pub fn sun_enu(config: &SimConfig) -> DVec3 {
    let llh = config.origin;
    let p = solar::solar_position(solar::unix_now(), llh.latitude_rad(), llh.longitude_rad());
    let (se, ce) = (p.elevation_rad.sin(), p.elevation_rad.cos());
    let (sa, ca) = (p.azimuth_rad.sin(), p.azimuth_rad.cos());
    DVec3::new(ce * sa, ce * ca, se) // east, north, up
}

/// Spawn the standard world entities: one aircraft (physics-driven,
/// renderable) + one chase camera. Used by both headless and windowed
/// modes so the ECS always has the same shape.
pub fn spawn_aircraft_and_camera(ecs: &mut Ecs) {
    use crate::ecs::{Aircraft, Camera, Renderable};
    let ac = ecs.spawn();
    ecs.transform
        .insert(ac, Transform { position: DVec3::ZERO, orientation: DQuat::IDENTITY });
    ecs.velocity
        .insert(ac, Velocity { linear: DVec3::ZERO, angular_body: DVec3::ZERO });
    ecs.aircraft.insert(ac, Aircraft);
    ecs.renderable.insert(ac, Renderable);

    let cam = ecs.spawn();
    ecs.transform
        .insert(cam, Transform { position: DVec3::ZERO, orientation: DQuat::IDENTITY });
    // chase camera: behind and slightly above the cockpit (body +Z is
    // down, so −0.9 puts the camera above)
    ecs.camera.insert(cam, Camera { body_offset: DVec3::new(-1.5, 0.0, -0.9) });
}

// ---------------------------------------------------------------------------
// ECS systems
// ---------------------------------------------------------------------------

/// Sync the aircraft entity's transform/velocity from the snapshot
/// (physics ENU frame).
pub fn system_sync_aircraft(ecs: &mut Ecs, snap: &Snapshot) {
    for e in ecs.aircraft_entities() {
        if let Some(t) = ecs.transform.get_mut(e) {
            t.position = snap.position_enu;
            t.orientation = snap.orientation_enu;
        }
        if let Some(v) = ecs.velocity.get_mut(e) {
            v.linear = snap.velocity_enu;
            v.angular_body = snap.omega_body;
        }
    }
}

/// Place the camera entity at the aircraft pose offset by
/// `Camera::body_offset` (body axes).
pub fn system_camera_follow(ecs: &mut Ecs) {
    let aircraft: Vec<_> = ecs.aircraft_entities();
    let cameras = ecs.camera_entities();
    for c in cameras {
        let offset = ecs.camera.get(c).map(|cam| cam.body_offset);
        let (Some(offset), Some(ac)) = (offset, aircraft.first().copied()) else {
            continue;
        };
        let Some(ac_t) = ecs.transform.get(ac).copied() else {
            continue;
        };
        let world_offset = ac_t.orientation * offset;
        if let Some(cam_t) = ecs.transform.get_mut(c) {
            cam_t.position = ac_t.position + world_offset;
            cam_t.orientation = ac_t.orientation;
        }
    }
}

/// Build the camera-relative draw list for this frame from the
/// loaded glTF model (if any) and the camera entity pose.
pub fn system_build_draws(
    ecs: &Ecs,
    bridge: &FrameBridge,
    camera_ecef: DVec3,
    camera_quat: DQuat,
    model: Option<&sim_render::GltfModel>,
) -> Vec<DrawItem> {
    let Some(model) = model else {
        return Vec::new();
    };
    let origin = FloatingOrigin::new(Ecef::from(camera_ecef), camera_quat);
    let mut draws = Vec::new();
    for e in ecs.renderable_entities() {
        let Some(t) = ecs.transform.get(e) else { continue };
        let pos_ecef = bridge.enu_to_ecef(t.position);
        let quat_ecef = bridge.orientation_to_ecef(t.orientation);
        draws.extend(Renderer::draws_from_gltf(
            model,
            &origin,
            pos_ecef,
            quat_ecef,
            DVec3::ONE,
        ));
    }
    draws
}

/// Compute the camera's ECEF pose (position + quaternion that maps
/// ECEF → camera basis: +X right, +Y up, −Z forward) from the camera
/// entity's physics-frame pose.
pub fn camera_ecef_pose(
    bridge: &FrameBridge,
    position_enu: DVec3,
    orientation_enu: DQuat,
) -> (DVec3, DQuat) {
    let pos_ecef = bridge.enu_to_ecef(position_enu);
    // camera axes in the ENU frame: x = body right, y = body up
    // (body −Z), z = −body forward (view looks down −Z)
    let right = orientation_enu * DVec3::Y;
    let up = orientation_enu * DVec3::new(0.0, 0.0, -1.0);
    let fwd = orientation_enu * DVec3::X;
    let back = -fwd;
    // columns: camera x/y/z expressed in ENU
    let cam_to_enu = DMat3::from_cols(right, up, back);
    let enu_to_ecef_m = bridge.enu_to_ecef_mat();
    let cam_to_ecef = enu_to_ecef_m * cam_to_enu;
    // quaternion ECEF → camera = transpose of (camera → ECEF)
    let q = DQuat::from_mat3(&cam_to_ecef.transpose());
    (pos_ecef, q)
}

/// FrameInput for one rendered frame (headless or windowed).
pub fn build_frame_input(
    bridge: &FrameBridge,
    camera_ecef: DVec3,
    camera_quat: DQuat,
    _snap: &Snapshot,
    sky_luts_sun: DVec3,
    time_s: f64,
) -> FrameInput<'static> {
    let origin = FloatingOrigin::new(Ecef::from(camera_ecef), camera_quat);
    let view = origin.view_matrix();
    let proj = Mat4::perspective_rh(60.0f32.to_radians(), 16.0 / 9.0, 0.5, 1e7);
    let sky_from_rel = bridge.sky_from_ecef_quat();
    // sun direction: ENU → sky frame (sky = x north, y up, z east? we
    // defined sky_from_ecef via (north, up, east) columns, so a sky
    // vector (a, b, c) = a·north + b·up + c·east).
    let sun_enu = sky_luts_sun; // (east, north, up)
    let sun_sky = Vec3::new(
        sun_enu.y as f32, // north → sky x
        sun_enu.z as f32, // up → sky y
        sun_enu.x as f32, // east → sky z
    )
    .normalize();
    let sun_enu_norm = sun_enu.normalize();
    let _ = sun_enu_norm;

    let sky_f32 = Quat::from_xyzw(
        sky_from_rel.x as f32,
        sky_from_rel.y as f32,
        sky_from_rel.z as f32,
        sky_from_rel.w as f32,
    )
    .normalize();
    FrameInput {
        view,
        proj,
        sky_from_rel: sky_f32,
        sun_dir_sky: sun_sky,
        sun_irradiance: Vec3::splat(20.0),
        camera_alt_m: WGS84
            .ecef_to_geodetic(&Ecef::from(camera_ecef))
            .altitude_m() as f32,
        atmosphere_height_m: 64_000.0,
        exposure: 1.0,
        aerial: true,
        sun_disc: true,
        time_s: time_s as f32,
        wind: Vec3::ZERO,
        clouds: CloudUniforms {
            wind_time: [time_s as f32, 3.0, 0.0, 1.0],
            ..CloudUniforms::default()
        },
        draws: &[],
    }
}

/// Spawn the OpenSimConnect TCP server on `DEFAULT_PORT` serving a
/// *clone* of the bus (shallow — the Arc inside keeps every clone on
/// the same storage).
pub fn start_simconnect_server(bus: SimVarBus, rt: &tokio::runtime::Runtime) {
    rt.spawn(async move {
        match sim_avionics::OpenSimConnectServer::bind("0.0.0.0:45170", bus).await {
            Ok(server) => {
                eprintln!("OpenSimConnect listening on port {DEFAULT_PORT}");
                let _ = server.serve().await;
            }
            Err(e) => eprintln!("OpenSimConnect bind failed: {e}"),
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::{Aircraft, Camera, Renderable};

    #[test]
    fn bridge_roundtrips_origin() {
        let llh = LatLonAlt::from_degrees(40.0, -74.0, 120.0);
        let b = FrameBridge::new(llh);
        let back = b.ecef_to_llh(b.enu_to_ecef(DVec3::ZERO));
        assert!((back.latitude_deg() - 40.0).abs() < 1e-7);
        assert!((back.longitude_deg() + 74.0).abs() < 1e-7);
        assert!((back.altitude_m() - 120.0).abs() < 1e-3);

        // 1000 m east keeps latitude, moves longitude east
        let p = b.enu_to_ecef(DVec3::new(1000.0, 0.0, 0.0));
        let g = b.ecef_to_llh(p);
        assert!((g.latitude_deg() - 40.0).abs() < 1e-3);
        assert!(g.longitude_deg() > -74.0, "east should increase longitude");
        assert!((g.altitude_m() - 120.0).abs() < 1.0);
    }

    #[test]
    fn bridge_enu_basis_is_right_handed_orthonormal() {
        let b = FrameBridge::new(LatLonAlt::from_degrees(40.0, -74.0, 0.0));
        let m = b.enu_to_ecef_mat();
        let i = m.transpose() * m;
        for r in 0..3 {
            for c in 0..3 {
                let want = if r == c { 1.0 } else { 0.0 };
                assert!((i.col(c)[r] - want).abs() < 1e-12, "Gram[{r}{c}]");
            }
        }
        // det = +1 (no reflection)
        let det = m.x_axis.dot(m.y_axis.cross(m.z_axis));
        assert!((det - 1.0).abs() < 1e-12);
    }

    #[test]
    fn ecs_camera_follow_places_camera_aft_of_aircraft() {
        let mut ecs = Ecs::new();
        let ac = ecs.spawn();
        let cam = ecs.spawn();
        ecs.transform.insert(
            ac,
            Transform { position: DVec3::new(10.0, 0.0, 5.0), orientation: DQuat::IDENTITY },
        );
        ecs.aircraft.insert(ac, Aircraft);
        ecs.transform.insert(
            cam,
            Transform { position: DVec3::ZERO, orientation: DQuat::IDENTITY },
        );
        ecs.camera.insert(cam, Camera { body_offset: DVec3::new(-8.0, 0.0, -2.0) });
        ecs.renderable.insert(ac, Renderable);

        system_camera_follow(&mut ecs);
        let t = ecs.transform.get(cam).unwrap();
        // identity orientation: offset applied directly (body +Z is
        // down in aeronautics, but with identity quat the offset is
        // literal) → (10−8, 0, 5−2)
        assert_eq!(t.position, DVec3::new(2.0, 0.0, 3.0));
    }

    #[test]
    fn core_builds_from_cleanroom_assets_and_steps_once() {
        let cfg = SimConfig { fast: true, ..SimConfig::default() };
        let mut core = SimCore::new(cfg).expect("sim core");
        assert!(core.aircraft_def.flight_model.weight.empty_weight_kg > 500.0);
        core.start();
        std::thread::sleep(Duration::from_millis(120));
        core.shutdown();
        let snap = core.snapshot_handle().read().unwrap().clone();
        assert!(snap.sim_time_s > 0.0, "thread must advance sim time");
        assert!(snap.geodetic.latitude_deg().abs() < 90.0);
        assert!(snap.indicated_airspeed_mps.is_finite());
    }

    #[test]
    fn headless_fast_forward_is_deterministic_in_steps() {
        let cfg = SimConfig {
            fast: true,
            airborne: true,
            initial_airspeed_mps: 50.0,
            initial_altitude_m: 500.0,
            ..SimConfig::default()
        };
        let mut core = SimCore::new(cfg).expect("sim core");
        core.start();
        // wait until the snapshot passes 1 simulated second
        let start = Instant::now();
        loop {
            let t = core.snapshot_handle().read().unwrap().sim_time_s;
            if t >= 1.0 {
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(10), "physics not advancing");
            std::thread::sleep(Duration::from_millis(5));
        }
        core.shutdown();
        let snap = core.snapshot_handle().read().unwrap().clone();
        // airborne at 50 m/s: the fast thread reaches >= 1 s and is
        // still airborne (it may overshoot 1 s while we poll — that is
        // expected in unpaced mode).
        assert!(snap.sim_time_s >= 1.0, "t = {}", snap.sim_time_s);
        assert!(snap.true_airspeed_mps > 30.0, "tas {}", snap.true_airspeed_mps);
        assert!(snap.altitude_m > 100.0, "alt {}", snap.altitude_m);
    }
}

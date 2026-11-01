//! OpenMSFS main binary.
//!
//! Modes
//! -----
//! * `--seconds N` (default 10): **headless** — runs the 1000 Hz
//!   physics thread as fast as possible for N simulated seconds,
//!   prints one telemetry line per simulated second and exits 0.
//!   This is the mode CI/sandboxes use (no display required).
//! * `--windowed`: opens a winit window and presents sky + clouds +
//!   optional glTF model through [`sim_render::Renderer`].
//! * `--realtime`: pace headless physics against the wall clock;
//!   windowed mode is always real-time.
//! * `--simconnect`: serve the OpenSimConnect TCP protocol on port
//!   45170 (shares the SimVarBus with the physics thread).
//! * `--model PATH.glb`: load an aircraft glTF/GLB and draw it at the
//!   aircraft pose (clean-room assets only — bring your own model).
//! * `--ap-heading DEG` / `--ap-alt FT`: engage the autopilot cascade.
//! * `--origin LAT,LON,ALT` / `--airborne`: initial conditions.
//!
//! ECS: [`ecs::Ecs`] carries the aircraft + chase-camera entities;
//! systems run per rendered frame (and once per telemetry tick
//! headless). The physics thread itself is independent of
//! presentation — it only shares a `Snapshot` and a `ControlCommand`.

use glam::{Mat4, Quat, Vec3};
use open_msfs::app;
use open_msfs::ecs::Ecs;
use open_msfs::sim::{
    spawn_aircraft_and_camera, start_simconnect_server, system_build_draws, system_camera_follow,
    system_sync_aircraft, ControlCommand, SimConfig, SimCore, Snapshot,
};
use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use wgpu::TextureFormat;

#[derive(Debug, Clone, PartialEq)]
struct Cli {
    mode: Mode,
    seconds: f64,
    realtime: bool,
    simconnect: bool,
    model: Option<String>,
    world_data: Option<String>,
    origin: Option<(f64, f64, f64)>,
    airborne: bool,
    ground: bool,
    heading_deg: Option<f64>,
    ap_heading: Option<f64>,
    ap_altitude_ft: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Headless,
    Windowed,
}

fn physics_fast_forward(mode: Mode, realtime_requested: bool) -> bool {
    mode == Mode::Headless && !realtime_requested
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            mode: Mode::Headless,
            seconds: 10.0,
            realtime: false,
            simconnect: false,
            model: None,
            world_data: None,
            origin: None,
            airborne: false,
            ground: false,
            heading_deg: None,
            ap_heading: None,
            ap_altitude_ft: None,
        }
    }
}

fn print_usage() {
    println!(
        "OpenMSFS — clean-room flight simulator\n\
         \n\
         usage: open-msfs [options]\n\
         \n\
         modes:\n\
           --seconds N          headless simulation length in sim-seconds [10]\n\
           --windowed           run the winit/wgpu window (needs a display + GPU)\n\
         \n\
         simulation:\n\
           --realtime           pace headless physics to wall clock\n\
           --airborne           start 1,000 m above field at 55 m/s\n\
           --ground             start on a runway instead of the NYC flight start\n\
           --origin LAT,LON,ALT field origin in degrees/metres [40,-74,0]\n\
           --heading DEG        initial heading for ground or airborne starts\n\
           --ap-heading DEG     engage heading hold autopilot\n\
           --ap-alt FT          engage altitude hold autopilot\n\
           --tz MINUTES         civil UTC offset for the sun [330]\n\
           --model PATH.glb     aircraft model to render at the pose\n\
           --world-data PATH    local sim-world tile root (loads origin DEM)\n\
           --simconnect         serve OpenSimConnect on port 45170\n\
           --help               show this help"
    );
}

fn parse_cli(args: &[String]) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{a} needs a value"))
        };
        match a.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--windowed" => cli.mode = Mode::Windowed,
            "--headless" => cli.mode = Mode::Headless,
            "--realtime" => cli.realtime = true,
            "--airborne" => cli.airborne = true,
            "--ground" => cli.ground = true,
            "--simconnect" => cli.simconnect = true,
            "--seconds" => {
                cli.seconds = next(&mut i)?
                    .parse()
                    .map_err(|e| format!("--seconds: {e}"))?
            }
            "--model" => cli.model = Some(next(&mut i)?),
            "--world-data" => cli.world_data = Some(next(&mut i)?),
            "--heading" => {
                cli.heading_deg = Some(
                    next(&mut i)?
                        .parse()
                        .map_err(|e| format!("--heading: {e}"))?,
                )
            }
            "--ap-heading" => {
                cli.ap_heading = Some(
                    next(&mut i)?
                        .parse()
                        .map_err(|e| format!("--ap-heading: {e}"))?,
                )
            }
            "--ap-alt" => {
                cli.ap_altitude_ft = Some(
                    next(&mut i)?
                        .parse()
                        .map_err(|e| format!("--ap-alt: {e}"))?,
                )
            }
            "--origin" => {
                let raw = next(&mut i)?;
                let parts: Vec<f64> = raw
                    .split(',')
                    .map(|p| {
                        p.trim()
                            .parse::<f64>()
                            .map_err(|e| format!("--origin: {e}"))
                    })
                    .collect::<Result<_, _>>()?;
                if parts.len() != 3 {
                    return Err(format!("--origin expects LAT,LON,ALT — got {raw:?}"));
                }
                cli.origin = Some((parts[0], parts[1], parts[2]));
            }
            other => return Err(format!("unknown argument {other:?} (try --help)")),
        }
        i += 1;
    }
    if cli.airborne && cli.ground {
        return Err("--airborne and --ground cannot be used together".into());
    }
    Ok(cli)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_cli(&args) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let mut config = SimConfig {
        airborne: cli.airborne,
        fast: physics_fast_forward(cli.mode, cli.realtime),
        ..SimConfig::default()
    };
    if cli.airborne {
        config.initial_altitude_m = 1_000.0;
        config.initial_airspeed_mps = 55.0;
    }
    // A ground-level spawn at the default Lower Manhattan coordinate can
    // intersect buildings. Give the normal windowed experience clear air
    // around the aircraft; --ground keeps the runway/ground-start option.
    let default_nyc_flight =
        cli.mode == Mode::Windowed && !cli.ground && !cli.airborne && cli.origin.is_none();
    if default_nyc_flight {
        config.airborne = true;
        config.initial_altitude_m = 1_000.0;
        config.initial_airspeed_mps = 55.0;
    }
    if cli.ground && cli.origin.is_none() {
        // Start 150 m into KJFK runway 04R (OSM regional surface bundle).
        config.origin = sim_geo::LatLonAlt::from_degrees(40.6267, -73.7693, 4.0);
        config.initial_heading_rad = 30.7_f64.to_radians();
    }
    if let Some((lat, lon, alt)) = cli.origin {
        config.origin = sim_geo::LatLonAlt::from_degrees(lat, lon, alt);
    }
    if let Some(heading) = cli.heading_deg {
        config.initial_heading_rad = heading.rem_euclid(360.0).to_radians();
    }

    let default_world_data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/world");
    let world_data = cli
        .world_data
        .as_ref()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            (cli.mode == Mode::Windowed && default_world_data.is_dir())
                .then_some(default_world_data)
        });

    let terrain = match world_data.as_deref() {
        Some(root) if cli.mode == Mode::Windowed => {
            let asset = app::load_terrain_tile(root, config.origin)?;
            println!(
                "terrain: DEM face={} level={} x={} y={} ({}x{} samples), field {:.1}m",
                asset.mesh.id.face,
                asset.mesh.id.level,
                asset.mesh.id.x,
                asset.mesh.id.y,
                asset.mesh.width,
                asset.mesh.height,
                asset.field_elevation_m,
            );
            config.origin = sim_geo::LatLonAlt::new(
                config.origin.latitude_rad(),
                config.origin.longitude_rad(),
                asset.field_elevation_m,
            );
            Some(asset)
        }
        Some(_) if cli.world_data.is_some() => {
            eprintln!(
                "--world-data is currently used by --windowed mode; ignoring in headless mode"
            );
            None
        }
        _ => None,
    };

    let mut core = SimCore::new(config.clone())?;
    let snapshot = core.snapshot_handle();
    let command = core.command_handle();
    let bus = core.bus();

    // Use the detailed bundled C172 by default in windowed mode; a user
    // supplied path still overrides it.
    let bundled_model =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/aircraft/c172.glb");
    let model_path = cli
        .model
        .as_ref()
        .map(std::path::PathBuf::from)
        .or_else(|| {
            (cli.mode == Mode::Windowed && bundled_model.is_file()).then_some(bundled_model)
        });
    let model = match model_path {
        Some(path) => {
            let mut m = sim_render::GltfSource::from_file(&path)
                .map_err(|e| format!("model {path:?}: {e}"))?;
            if path
                .file_name()
                .is_some_and(|name| name.eq_ignore_ascii_case("c172.glb"))
            {
                let (length, wingspan, height) = align_c172_model(&mut m);
                println!("C172 model: {length:.2}m long, {wingspan:.2}m span, {height:.2}m high");
            }
            println!(
                "model: {} mesh(es), {} material(s), {} instance(s)",
                m.meshes.len(),
                m.materials.len(),
                m.instances.len()
            );
            Some(m)
        }
        None => None,
    };
    // autopilot engagement from the CLI
    {
        let mut guard = command.lock().unwrap_or_else(|e| e.into_inner());
        if default_nyc_flight {
            guard.throttle = 0.65;
            guard.autopilot = Some(sim_avionics::ApMode::AltitudeHold);
            guard.ap_altitude_ft = Some(config.initial_altitude_m * 3.280_839_895);
        }
        if let Some(h) = cli.ap_heading {
            guard.autopilot = Some(sim_avionics::ApMode::HeadingHold);
            guard.ap_heading_deg = Some(h);
            println!("autopilot: HEADING HOLD {h}°");
        }
        if let Some(a) = cli.ap_altitude_ft {
            guard.autopilot = Some(sim_avionics::ApMode::AltitudeHold);
            guard.ap_altitude_ft = Some(a);
            println!("autopilot: ALTITUDE HOLD {a} ft");
        }
    }

    if cli.mode == Mode::Windowed {
        println!(
            "controls: W/S pitch, A/D roll, Q/E rudder, R/F throttle, B brake, G flaps, P autopilot/manual, Esc quit"
        );
    }

    // OpenSimConnect (shares the bus — clones are shallow)
    let tokio_rt = if cli.simconnect {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| format!("tokio: {e}"))?;
        start_simconnect_server(bus.clone(), &rt);
        Some(rt)
    } else {
        None
    };

    let mut ecs = Ecs::new();
    spawn_aircraft_and_camera(&mut ecs);

    core.start();

    match cli.mode {
        Mode::Headless => {
            let result = run_headless(
                &cli,
                &mut core,
                &mut ecs,
                &snapshot,
                &command,
                model.as_ref(),
            );
            core.shutdown();
            drop(tokio_rt);
            result
        }
        Mode::Windowed => {
            let sky_sun = open_msfs::sim::sun_enu(&config);
            let app = app::App::new(
                app::WindowInput {
                    command: Arc::clone(&command),
                    snapshot: Arc::clone(&snapshot),
                },
                &core,
                sky_sun,
                model,
                terrain,
                ecs,
            );
            let result = app.run();
            core.shutdown();
            drop(tokio_rt);
            result
        }
    }
}

/// Convert the bundled glTF Y-up aircraft frame to the simulator body
/// frame (+X forward, +Y right, +Z down), center it at its CG, and place
/// its landing wheels one metre below the root.
fn align_c172_model(model: &mut sim_render::GltfModel) -> (f32, f32, f32) {
    // The source C172 faces toward glTF -X. Map its Y-up frame to the
    // simulator's Z-down frame, then turn its nose into body +X.
    let gltf_to_body = Quat::from_rotation_z(std::f32::consts::PI)
        * Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    for instance in &model.instances {
        if let Some(mesh) = model.meshes.get(instance.mesh) {
            for vertex in &mesh.vertices {
                let point = gltf_to_body
                    * instance
                        .world
                        .transform_point3(Vec3::from_array(vertex.position));
                min = min.min(point);
                max = max.max(point);
            }
        }
    }
    let size = max - min;
    let offset = Vec3::new(-(min.x + max.x) * 0.5, -(min.y + max.y) * 0.5, 1.0 - max.z);
    let conversion = Mat4::from_translation(offset) * Mat4::from_quat(gltf_to_body);
    for instance in &mut model.instances {
        instance.world = conversion * instance.world;
    }
    (size.x, size.y, size.z)
}

fn run_headless(
    cli: &Cli,
    core: &mut SimCore,
    ecs: &mut Ecs,
    snapshot: &Arc<RwLock<Snapshot>>,
    command: &Arc<Mutex<ControlCommand>>,
    model: Option<&sim_render::GltfModel>,
) -> Result<(), String> {
    let start = Instant::now();
    let mut last_reported = -1i64;
    println!(
        "{:>8}  {:>10}  {:>11}  {:>7}  {:>6}  {:>5}  {:>6}",
        "t(s)", "lat", "lon", "alt(m)", "ias", "hdg", "pitch"
    );
    loop {
        let snap = snapshot.read().unwrap_or_else(|e| e.into_inner()).clone();

        // ECS systems run on every telemetry tick too — the same
        // pipeline the windowed renderer uses.
        system_sync_aircraft(ecs, &snap);
        system_camera_follow(ecs);

        let sec = snap.sim_time_s.floor() as i64;
        if sec != last_reported {
            last_reported = sec;
            println!(
                "{:>8.1}  {:>10.5}  {:>11.5}  {:>7.1}  {:>6.1}  {:>6.1}  {:>6.1}{}",
                snap.sim_time_s,
                snap.geodetic.latitude_deg(),
                snap.geodetic.longitude_deg(),
                snap.altitude_m,
                snap.indicated_airspeed_mps * 1.943_844,
                snap.heading_deg,
                snap.pitch_deg,
                if snap.gear_on_ground {
                    "  [ground]"
                } else {
                    ""
                }
            );
        }

        if snap.sim_time_s >= cli.seconds {
            // demonstrate the render-side systems once (draw list
            // construction with the sanctioned f64→f32 rebasing)
            if let Some(model) = model {
                let cam = ecs.camera_entities().first().copied();
                let (cp, cq) = cam
                    .and_then(|c| ecs.transform.get(c).map(|t| (t.position, t.orientation)))
                    .unwrap_or((snap.position_enu, snap.orientation_enu));
                let cam_ecef = core.bridge.enu_to_ecef(cp);
                let cam_quat = core.bridge.orientation_to_ecef(cq);
                let draws = system_build_draws(ecs, &core.bridge, cam_ecef, cam_quat, Some(model));
                println!(
                    "final state: t={:.2}s lat={:.5} lon={:.5} alt={:.1}m ias={:.1}kt \
                     heading={:.1}° bank={:.1}° draws={} diverged={}",
                    snap.sim_time_s,
                    snap.geodetic.latitude_deg(),
                    snap.geodetic.longitude_deg(),
                    snap.altitude_m,
                    snap.indicated_airspeed_mps * 1.943_844,
                    snap.heading_deg,
                    snap.roll_deg,
                    draws.len(),
                    snap.diagnostics.diverged,
                );
            } else {
                println!(
                    "final state: t={:.2}s lat={:.5} lon={:.5} alt={:.1}m ias={:.1}kt \
                     heading={:.1}° bank={:.1}° diverged={}",
                    snap.sim_time_s,
                    snap.geodetic.latitude_deg(),
                    snap.geodetic.longitude_deg(),
                    snap.altitude_m,
                    snap.indicated_airspeed_mps * 1.943_844,
                    snap.heading_deg,
                    snap.roll_deg,
                    snap.diagnostics.diverged,
                );
            }
            if snap.diagnostics.diverged {
                return Err("physics diverged".into());
            }
            break;
        }

        if start.elapsed() > Duration::from_secs(cli.seconds as u64 * 30 + 60) {
            return Err(format!(
                "timed out waiting for sim_time {:.1}s (at {:.3}s)",
                cli.seconds, snap.sim_time_s
            ));
        }
        std::thread::sleep(Duration::from_millis(2));
    }

    let _ = command;
    Ok(())
}

/// Windowed mode needs a swapchain format check helper used by tests.
#[allow(dead_code)]
fn tonemap_target_format() -> TextureFormat {
    TextureFormat::Bgra8UnormSrgb
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn cli_parses_defaults() {
        let cli = parse_cli(&[]).unwrap();
        assert_eq!(cli.mode, Mode::Headless);
        assert_eq!(cli.seconds, 10.0);
        assert!(!cli.realtime);
        assert!(!cli.simconnect);
    }

    #[test]
    fn windowed_physics_is_real_time_by_default() {
        assert!(physics_fast_forward(Mode::Headless, false));
        assert!(!physics_fast_forward(Mode::Headless, true));
        assert!(!physics_fast_forward(Mode::Windowed, false));
        assert!(!physics_fast_forward(Mode::Windowed, true));
    }

    #[test]
    fn detailed_c172_asset_loads_and_aligns_to_the_physics_body() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/aircraft/c172.glb");
        let mut model = sim_render::GltfSource::from_file(&path).expect("bundled C172 GLB");
        assert!(model.meshes.len() > 100);
        assert!(model.materials.len() >= 8);
        let (length, span, height) = align_c172_model(&mut model);
        assert!((length - 8.23).abs() < 0.1);
        assert!((span - 11.32).abs() < 0.1);
        assert!((height - 2.46).abs() < 0.1);
        let position = |name: &str| {
            model
                .instances
                .iter()
                .find(|instance| instance.node_name.as_deref() == Some(name))
                .expect("named model part")
                .world
                .transform_point3(Vec3::ZERO)
                .x
        };
        assert!(
            position("Propeller") > position("rudder_1"),
            "model nose must point along the simulator's +X forward axis"
        );
    }

    #[test]
    fn cli_parses_full_invocation() {
        let cli = parse_cli(&args(&[
            "--windowed",
            "--realtime",
            "--seconds",
            "25.5",
            "--origin",
            "51.5,-0.12,45",
            "--heading",
            "31",
            "--airborne",
            "--ap-heading",
            "270",
            "--ap-alt",
            "3500",
            "--model",
            "assets/plane.glb",
            "--world-data",
            "assets/world",
            "--simconnect",
        ]))
        .unwrap();
        assert_eq!(cli.mode, Mode::Windowed);
        assert!(cli.realtime);
        assert_eq!(cli.seconds, 25.5);
        assert_eq!(cli.origin, Some((51.5, -0.12, 45.0)));
        assert_eq!(cli.heading_deg, Some(31.0));
        assert!(cli.airborne);
        assert!(!cli.ground);
        assert_eq!(cli.ap_heading, Some(270.0));
        assert_eq!(cli.ap_altitude_ft, Some(3500.0));
        assert_eq!(cli.model.as_deref(), Some("assets/plane.glb"));
        assert_eq!(cli.world_data.as_deref(), Some("assets/world"));
        assert!(cli.simconnect);
    }

    #[test]
    fn ground_start_is_explicit_and_conflicts_with_airborne() {
        assert!(
            parse_cli(&args(&["--windowed", "--ground"]))
                .unwrap()
                .ground
        );
        assert!(parse_cli(&args(&["--ground", "--airborne"])).is_err());
    }

    #[test]
    fn cli_rejects_bad_input() {
        assert!(parse_cli(&args(&["--nope"])).is_err());
        assert!(parse_cli(&args(&["--seconds"])).is_err());
        assert!(parse_cli(&args(&["--origin", "1,2"])).is_err());
        assert!(parse_cli(&args(&["--origin", "a,b,c"])).is_err());
        assert!(parse_cli(&args(&["--heading", "north"])).is_err());
    }

    #[test]
    fn headless_run_completes_and_reports() {
        let cli = Cli {
            seconds: 1.0,
            realtime: false,
            ..Cli::default()
        };
        let config = SimConfig {
            fast: true,
            ..SimConfig::default()
        };
        let mut core = SimCore::new(config).expect("core");
        let snapshot = core.snapshot_handle();
        let command = core.command_handle();
        let mut ecs = Ecs::new();
        spawn_aircraft_and_camera(&mut ecs);
        core.start();
        let res = run_headless(&cli, &mut core, &mut ecs, &snapshot, &command, None);
        core.shutdown();
        assert!(res.is_ok(), "{res:?}");
    }

    #[test]
    fn headless_airborne_run_stays_airborne() {
        let cli = Cli {
            seconds: 2.0,
            realtime: false,
            airborne: true,
            ..Cli::default()
        };
        let config = SimConfig {
            fast: true,
            airborne: true,
            initial_airspeed_mps: 55.0,
            initial_altitude_m: 500.0,
            ..SimConfig::default()
        };
        let mut core = SimCore::new(config).expect("core");
        let snapshot = core.snapshot_handle();
        let command = core.command_handle();
        let mut ecs = Ecs::new();
        spawn_aircraft_and_camera(&mut ecs);
        core.start();
        let res = run_headless(&cli, &mut core, &mut ecs, &snapshot, &command, None);
        core.shutdown();
        assert!(res.is_ok(), "{res:?}");
        let snap = snapshot.read().unwrap().clone();
        assert!(snap.sim_time_s >= 2.0);
        assert!(
            snap.altitude_m > 400.0,
            "should stay near release altitude: {}",
            snap.altitude_m
        );
    }
}

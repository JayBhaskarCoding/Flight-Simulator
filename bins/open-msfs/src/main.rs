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
//! * `--realtime`: pace physics against the wall clock (both modes).
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

use open_msfs::app;
use open_msfs::ecs::Ecs;
use open_msfs::sim::{
    spawn_aircraft_and_camera, start_simconnect_server, system_build_draws, system_camera_follow,
    system_sync_aircraft, ControlCommand, SimConfig, SimCore, Snapshot,
};
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
    origin: Option<(f64, f64, f64)>,
    airborne: bool,
    ap_heading: Option<f64>,
    ap_altitude_ft: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Headless,
    Windowed,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            mode: Mode::Headless,
            seconds: 10.0,
            realtime: false,
            simconnect: false,
            model: None,
            origin: None,
            airborne: false,
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
           --realtime           pace the 1000 Hz physics thread to wall clock\n\
           --airborne           start airborne (--origin ALT metres above field)\n\
           --origin LAT,LON,ALT field origin in degrees/metres [40,-74,0]\n\
           --ap-heading DEG     engage heading hold autopilot\n\
           --ap-alt FT          engage altitude hold autopilot\n\
           --tz MINUTES         civil UTC offset for the sun [330]\n\
           --model PATH.glb     aircraft model to render at the pose\n\
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
            args.get(*i).cloned().ok_or_else(|| format!("{a} needs a value"))
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
            "--simconnect" => cli.simconnect = true,
            "--seconds" => {
                cli.seconds = next(&mut i)?.parse().map_err(|e| format!("--seconds: {e}"))?
            }
            "--model" => cli.model = Some(next(&mut i)?),
            "--ap-heading" => {
                cli.ap_heading =
                    Some(next(&mut i)?.parse().map_err(|e| format!("--ap-heading: {e}"))?)
            }
            "--ap-alt" => {
                cli.ap_altitude_ft =
                    Some(next(&mut i)?.parse().map_err(|e| format!("--ap-alt: {e}"))?)
            }
            "--origin" => {
                let raw = next(&mut i)?;
                let parts: Vec<f64> = raw
                    .split(',')
                    .map(|p| p.trim().parse::<f64>().map_err(|e| format!("--origin: {e}")))
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
        fast: !cli.realtime,
        ..SimConfig::default()
    };
    if let Some((lat, lon, alt)) = cli.origin {
        config.origin = sim_geo::LatLonAlt::from_degrees(lat, lon, alt);
        if cli.airborne {
            config.initial_altitude_m = alt;
            config.initial_airspeed_mps = 55.0;
        }
    }

    let mut core = SimCore::new(config.clone())?;
    let snapshot = core.snapshot_handle();
    let command = core.command_handle();
    let bus = core.bus();

    // optional glTF aircraft model
    let model = match &cli.model {
        Some(path) => {
            let m = sim_render::GltfSource::from_file(std::path::Path::new(path))
                .map_err(|e| format!("model {path:?}: {e}"))?;
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
                &cli, &mut core, &mut ecs, &snapshot, &command, model.as_ref(),
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
                ecs,
            );
            let result = app.run();
            core.shutdown();
            drop(tokio_rt);
            result
        }
    }
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
                if snap.gear_on_ground { "  [ground]" } else { "" }
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
                let draws =
                    system_build_draws(ecs, &core.bridge, cam_ecef, cam_quat, Some(model));
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
    fn cli_parses_full_invocation() {
        let cli = parse_cli(&args(&[
            "--windowed",
            "--realtime",
            "--seconds",
            "25.5",
            "--origin",
            "51.5,-0.12,45",
            "--airborne",
            "--ap-heading",
            "270",
            "--ap-alt",
            "3500",
            "--model",
            "assets/plane.glb",
            "--simconnect",
        ]))
        .unwrap();
        assert_eq!(cli.mode, Mode::Windowed);
        assert!(cli.realtime);
        assert_eq!(cli.seconds, 25.5);
        assert_eq!(cli.origin, Some((51.5, -0.12, 45.0)));
        assert!(cli.airborne);
        assert_eq!(cli.ap_heading, Some(270.0));
        assert_eq!(cli.ap_altitude_ft, Some(3500.0));
        assert_eq!(cli.model.as_deref(), Some("assets/plane.glb"));
        assert!(cli.simconnect);
    }

    #[test]
    fn cli_rejects_bad_input() {
        assert!(parse_cli(&args(&["--nope"])).is_err());
        assert!(parse_cli(&args(&["--seconds"])).is_err());
        assert!(parse_cli(&args(&["--origin", "1,2"])).is_err());
        assert!(parse_cli(&args(&["--origin", "a,b,c"])).is_err());
    }

    #[test]
    fn headless_run_completes_and_reports() {
        let cli = Cli { seconds: 1.0, realtime: false, ..Cli::default() };
        let config = SimConfig { fast: true, ..SimConfig::default() };
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
        let cli = Cli { seconds: 2.0, realtime: false, airborne: true, ..Cli::default() };
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
        assert!(snap.altitude_m > 400.0, "should stay near release altitude: {}", snap.altitude_m);
    }
}

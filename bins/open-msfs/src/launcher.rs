//! Packaged Windows entry point for the interactive New York simulator.
//!
//! The main executable intentionally defaults to headless mode for tests and
//! command-line use. This tiny launcher starts its sibling with the bundled
//! region and C172 model so a normal double-click opens the simulator window.

#![cfg_attr(windows, windows_subsystem = "windows")]

fn main() {
    if let Err(error) = launch() {
        if let Ok(executable) = std::env::current_exe() {
            if let Some(package) = executable.parent() {
                if let Ok(mut log) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(package.join("flight-simulator.log"))
                {
                    use std::io::Write;
                    let _ = writeln!(log, "Flight Simulator could not start: {error}");
                }
            }
        }
    }
}

fn launch() -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::OpenOptions;
    use std::process::{Command, Stdio};

    let launcher = std::env::current_exe()?;
    let package = launcher
        .parent()
        .ok_or("launcher executable has no parent directory")?;
    let simulator = package.join(simulator_filename(
        cfg!(windows) && package.join("open-msfs-updated.exe").is_file(),
    ));
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(package.join("flight-simulator.log"))?;
    let stderr = Stdio::from(log.try_clone()?);

    let mut command = Command::new(simulator);
    command
        .current_dir(package)
        .args(simulator_args())
        .stdout(Stdio::from(log))
        .stderr(stderr);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }

    command.spawn()?;
    Ok(())
}

fn simulator_filename(has_updated_package_binary: bool) -> &'static str {
    if cfg!(windows) && has_updated_package_binary {
        "open-msfs-updated.exe"
    } else if cfg!(windows) {
        "open-msfs.exe"
    } else {
        "open-msfs"
    }
}

fn simulator_args() -> [&'static str; 5] {
    [
        "--windowed",
        "--world-data",
        "world",
        "--model",
        "aircraft/c172.glb",
    ]
}

#[cfg(test)]
mod tests {
    #[test]
    fn packaged_launcher_uses_bundled_region_and_aircraft() {
        assert_eq!(
            super::simulator_args(),
            [
                "--windowed",
                "--world-data",
                "world",
                "--model",
                "aircraft/c172.glb"
            ]
        );
        // The packaged names are Windows-only; other targets always use
        // the plain simulator binary (see `simulator_filename`).
        let (updated, plain) = if cfg!(windows) {
            ("open-msfs-updated.exe", "open-msfs.exe")
        } else {
            ("open-msfs", "open-msfs")
        };
        assert_eq!(super::simulator_filename(true), updated);
        assert_eq!(super::simulator_filename(false), plain);
    }
}

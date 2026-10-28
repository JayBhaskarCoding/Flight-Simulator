//! Autopilot: cascade PID loops over [`ApSample`] snapshots.
//!
//! Outer loops (heading / altitude / vertical speed) command a
//! reference (bank or pitch angle); inner loops (roll / pitch) track
//! that reference with surface deflections. All outputs are clamped to
//! configured limits, and every mode transition resets the affected
//! loops so stale integrators cannot kick the controls.

use crate::pid::{Pid, PidConfig, PidGains};

/// Autopilot engagement mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApMode {
    /// Disengaged: neutral surfaces, all loops reset.
    Off,
    /// Hold a fixed bank angle.
    RollHold,
    /// Hold a magnetic heading (commands bank via the roll loop).
    HeadingHold,
    /// Hold a pitch attitude.
    PitchHold,
    /// Hold a pressure altitude (commands pitch via the pitch loop).
    AltitudeHold,
    /// Hold a vertical speed (commands pitch via the pitch loop).
    VsHold,
}

/// One snapshot of the flight state consumed by [`Autopilot::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApSample {
    pub heading_deg: f64,
    pub pitch_deg: f64,
    pub roll_deg: f64,
    pub altitude_ft: f64,
    pub vertical_speed_fpm: f64,
}

/// Surface commands and references produced by one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApOutput {
    /// Aileron command in `[-1, 1]`, positive = roll right.
    pub aileron: f64,
    /// Elevator command in `[-1, 1]`, positive = nose up.
    pub elevator: f64,
    /// Rudder command in `[-1, 1]`, positive = nose right.
    pub rudder: f64,
    /// Bank angle the inner roll loop is tracking right now (deg).
    pub roll_reference_deg: f64,
    /// Pitch angle the inner pitch loop is tracking right now (deg).
    pub pitch_reference_deg: f64,
    /// Active mode.
    pub mode: ApMode,
}

/// Loop tuning and structural limits.
#[derive(Debug, Clone, PartialEq)]
pub struct AutopilotConfig {
    /// Maximum commanded bank (deg), used by heading hold.
    pub max_bank_deg: f64,
    /// Maximum commanded pitch (deg), used by altitude / VS hold.
    pub max_pitch_deg: f64,
    /// Aileron output limits `[-limit, limit]`.
    pub aileron_limit: f64,
    /// Elevator output limits `[-limit, limit]`.
    pub elevator_limit: f64,
    /// Roll loop (aileron) gains.
    pub roll_gains: PidGains,
    /// Heading loop (commands bank) gains.
    pub heading_gains: PidGains,
    /// Pitch loop (elevator) gains.
    pub pitch_gains: PidGains,
    /// Altitude loop (commands pitch) gains.
    pub altitude_gains: PidGains,
    /// Vertical-speed loop (commands pitch) gains.
    pub vs_gains: PidGains,
}

impl Default for AutopilotConfig {
    fn default() -> Self {
        Self {
            max_bank_deg: 25.0,
            max_pitch_deg: 12.0,
            aileron_limit: 1.0,
            elevator_limit: 1.0,
            roll_gains: PidGains::new(0.02, 0.004, 0.005),
            heading_gains: PidGains::new(0.6, 0.02, 0.0),
            pitch_gains: PidGains::new(0.05, 0.01, 0.012),
            altitude_gains: PidGains::new(0.03, 0.004, 0.0),
            vs_gains: PidGains::new(0.012, 0.003, 0.0),
        }
    }
}

impl AutopilotConfig {
    fn roll_pid(&self) -> Pid {
        Pid::new(PidConfig::new(
            self.roll_gains,
            -self.aileron_limit,
            self.aileron_limit,
        )
        .with_derivative_on_measurement())
    }

    fn heading_pid(&self) -> Pid {
        // Heading uses derivative-free control: wrapping makes
        // d(heading)/dt discontinuous at north.
        Pid::new(PidConfig::new(
            self.heading_gains,
            -self.max_bank_deg,
            self.max_bank_deg,
        ))
    }

    fn pitch_pid(&self) -> Pid {
        Pid::new(PidConfig::new(
            self.pitch_gains,
            -self.elevator_limit,
            self.elevator_limit,
        )
        .with_derivative_on_measurement())
    }

    fn altitude_pid(&self) -> Pid {
        Pid::new(PidConfig::new(
            self.altitude_gains,
            -self.max_pitch_deg,
            self.max_pitch_deg,
        ))
    }

    fn vs_pid(&self) -> Pid {
        Pid::new(PidConfig::new(
            self.vs_gains,
            -self.max_pitch_deg,
            self.max_pitch_deg,
        ))
    }
}

/// Shortest signed difference `to - from` in degrees, wrapped to
/// `(-180, 180]`.
pub fn wrap180(from_deg: f64, to_deg: f64) -> f64 {
    let mut d = (to_deg - from_deg) % 360.0;
    if d > 180.0 {
        d -= 360.0;
    } else if d <= -180.0 {
        d += 360.0;
    }
    d
}

/// The autopilot computer.
pub struct Autopilot {
    config: AutopilotConfig,
    mode: ApMode,
    roll_pid: Pid,
    heading_pid: Pid,
    pitch_pid: Pid,
    altitude_pid: Pid,
    vs_pid: Pid,
    /// Captured/commanded references (deg / ft / fpm).
    roll_ref_deg: f64,
    heading_ref_deg: f64,
    pitch_ref_deg: f64,
    altitude_ref_ft: f64,
    vs_ref_fpm: f64,
    /// References actually handed to the inner loops on the last step.
    last_roll_cmd_deg: f64,
    last_pitch_cmd_deg: f64,
}

impl Autopilot {
    pub fn new(config: AutopilotConfig) -> Self {
        let roll_pid = config.roll_pid();
        let heading_pid = config.heading_pid();
        let pitch_pid = config.pitch_pid();
        let altitude_pid = config.altitude_pid();
        let vs_pid = config.vs_pid();
        Self {
            config,
            mode: ApMode::Off,
            roll_pid,
            heading_pid,
            pitch_pid,
            altitude_pid,
            vs_pid,
            roll_ref_deg: 0.0,
            heading_ref_deg: 0.0,
            pitch_ref_deg: 0.0,
            altitude_ref_ft: 0.0,
            vs_ref_fpm: 0.0,
            last_roll_cmd_deg: 0.0,
            last_pitch_cmd_deg: 0.0,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(AutopilotConfig::default())
    }

    pub fn mode(&self) -> ApMode {
        self.mode
    }

    pub fn config(&self) -> &AutopilotConfig {
        &self.config
    }

    /// Current references (whatever was captured or commanded last).
    pub fn heading_target_deg(&self) -> f64 {
        self.heading_ref_deg
    }

    pub fn altitude_target_ft(&self) -> f64 {
        self.altitude_ref_ft
    }

    pub fn vs_target_fpm(&self) -> f64 {
        self.vs_ref_fpm
    }

    /// Switch modes. The new mode captures its reference from `sample`
    /// (current attitude/altitude), and every PID is reset so no stale
    /// integral survives the transition.
    pub fn set_mode(&mut self, mode: ApMode, sample: &ApSample) {
        self.mode = mode;
        self.roll_pid.reset();
        self.heading_pid.reset();
        self.pitch_pid.reset();
        self.altitude_pid.reset();
        self.vs_pid.reset();
        self.last_roll_cmd_deg = 0.0;
        self.last_pitch_cmd_deg = 0.0;
        self.roll_ref_deg = sample.roll_deg;
        self.heading_ref_deg = sample.heading_deg.rem_euclid(360.0);
        self.pitch_ref_deg = sample.pitch_deg;
        self.altitude_ref_ft = sample.altitude_ft;
        self.vs_ref_fpm = sample.vertical_speed_fpm;
    }

    /// Override the captured heading target (deg, any wrap).
    pub fn set_heading_target(&mut self, heading_deg: f64) {
        self.heading_ref_deg = heading_deg.rem_euclid(360.0);
        self.heading_pid.reset();
    }

    /// Override the captured altitude target (ft).
    pub fn set_altitude_target(&mut self, altitude_ft: f64) {
        self.altitude_ref_ft = altitude_ft;
        self.altitude_pid.reset();
    }

    /// Override the captured vertical-speed target (ft/min).
    pub fn set_vs_target(&mut self, vs_fpm: f64) {
        self.vs_ref_fpm = vs_fpm;
        self.vs_pid.reset();
    }

    /// Override the captured bank target (deg).
    pub fn set_roll_target(&mut self, roll_deg: f64) {
        self.roll_ref_deg = roll_deg;
        self.roll_pid.reset();
    }

    /// Override the captured pitch target (deg).
    pub fn set_pitch_target(&mut self, pitch_deg: f64) {
        self.pitch_ref_deg = pitch_deg;
        self.pitch_pid.reset();
    }

    /// Advance one control cycle and return surface commands.
    pub fn step(&mut self, sample: &ApSample, dt: f64) -> ApOutput {
        if self.mode == ApMode::Off {
            self.last_roll_cmd_deg = 0.0;
            self.last_pitch_cmd_deg = 0.0;
            return ApOutput {
                aileron: 0.0,
                elevator: 0.0,
                rudder: 0.0,
                roll_reference_deg: 0.0,
                pitch_reference_deg: 0.0,
                mode: ApMode::Off,
            };
        }

        // ---- inner roll reference -------------------------------------
        let roll_cmd = if self.mode == ApMode::HeadingHold {
            // Wrap-aware heading error: feed the PID a setpoint
            // re-expressed around the measurement so the difference is
            // always the shortest turn.
            let target = sample.heading_deg + wrap180(sample.heading_deg, self.heading_ref_deg);
            self.heading_pid.step(target, sample.heading_deg, dt)
        } else {
            self.roll_ref_deg
        };
        let roll_cmd = roll_cmd.clamp(-self.config.max_bank_deg, self.config.max_bank_deg);
        self.last_roll_cmd_deg = roll_cmd;

        // ---- inner pitch reference ------------------------------------
        let pitch_cmd = match self.mode {
            ApMode::AltitudeHold => self.altitude_pid.step(self.altitude_ref_ft, sample.altitude_ft, dt),
            ApMode::VsHold => self.vs_pid.step(self.vs_ref_fpm, sample.vertical_speed_fpm, dt),
            _ => self.pitch_ref_deg,
        };
        let pitch_cmd = pitch_cmd.clamp(-self.config.max_pitch_deg, self.config.max_pitch_deg);
        self.last_pitch_cmd_deg = pitch_cmd;

        // ---- inner loops ----------------------------------------------
        let aileron = self.roll_pid.step(roll_cmd, sample.roll_deg, dt);
        let elevator = self.pitch_pid.step(pitch_cmd, sample.pitch_deg, dt);

        ApOutput {
            aileron,
            elevator,
            rudder: 0.0,
            roll_reference_deg: roll_cmd,
            pitch_reference_deg: pitch_cmd,
            mode: self.mode,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ApSample {
        ApSample {
            heading_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
            altitude_ft: 5_000.0,
            vertical_speed_fpm: 0.0,
        }
    }

    #[test]
    fn off_mode_is_neutral() {
        let mut ap = Autopilot::with_defaults();
        let out = ap.step(&sample(), 0.02);
        assert_eq!(out, ApOutput {
            aileron: 0.0,
            elevator: 0.0,
            rudder: 0.0,
            roll_reference_deg: 0.0,
            pitch_reference_deg: 0.0,
            mode: ApMode::Off,
        });
    }

    #[test]
    fn set_mode_captures_current_state() {
        let mut ap = Autopilot::with_defaults();
        let mut s = sample();
        s.heading_deg = 271.0;
        s.roll_deg = 7.5;
        s.altitude_ft = 8_200.0;
        ap.set_mode(ApMode::HeadingHold, &s);
        assert_eq!(ap.mode(), ApMode::HeadingHold);
        assert!((ap.heading_target_deg() - 271.0).abs() < 1e-9);
        assert!((ap.altitude_target_ft() - 8_200.0).abs() < 1e-9);
    }

    #[test]
    fn heading_hold_commands_bank_toward_target() {
        let mut ap = Autopilot::with_defaults();
        let s = sample(); // heading 0
        ap.set_mode(ApMode::HeadingHold, &s);
        ap.set_heading_target(90.0); // turn right
        let out = ap.step(&s, 0.02);
        assert!(out.roll_reference_deg > 0.0, "bank cmd {}", out.roll_reference_deg);
        assert!(out.aileron > 0.0, "aileron {}", out.aileron);

        ap.set_heading_target(270.0); // turn left (shortest way)
        let out = ap.step(&s, 0.02);
        assert!(out.roll_reference_deg < 0.0, "bank cmd {}", out.roll_reference_deg);
    }

    #[test]
    fn heading_wrap_shortest_path() {
        // from 350 the target 10 is +20 deg, not -340.
        let mut ap = Autopilot::with_defaults();
        let mut s = sample();
        s.heading_deg = 350.0;
        ap.set_mode(ApMode::HeadingHold, &s);
        ap.set_heading_target(10.0);
        let out = ap.step(&s, 0.02);
        assert!(out.roll_reference_deg > 0.0, "bank cmd {}", out.roll_reference_deg);
        assert!(out.roll_reference_deg <= ap.config().max_bank_deg);
    }

    #[test]
    fn bank_command_respects_max_bank() {
        let mut ap = Autopilot::with_defaults();
        let s = sample();
        ap.set_mode(ApMode::HeadingHold, &s);
        ap.set_heading_target(180.0);
        // run many steps with a frozen aircraft: heading never moves
        let mut out = ap.step(&s, 0.02);
        for _ in 0..500 {
            out = ap.step(&s, 0.02);
        }
        assert!(out.roll_reference_deg <= ap.config().max_bank_deg + 1e-9);
        assert!(out.roll_reference_deg >= -ap.config().max_bank_deg - 1e-9);
        assert!(out.aileron.abs() <= 1.0);
    }

    #[test]
    fn altitude_hold_commands_pitch_up_when_below_target() {
        let mut ap = Autopilot::with_defaults();
        let mut s = sample();
        s.altitude_ft = 5_000.0;
        ap.set_mode(ApMode::AltitudeHold, &s);
        ap.set_altitude_target(6_000.0); // 1000 ft to climb
        let out = ap.step(&s, 0.05);
        assert!(out.pitch_reference_deg > 0.0, "pitch cmd {}", out.pitch_reference_deg);
        assert!(out.elevator > 0.0, "elevator {}", out.elevator);

        ap.set_altitude_target(4_000.0);
        let out = ap.step(&s, 0.05);
        assert!(out.pitch_reference_deg < 0.0, "pitch cmd {}", out.pitch_reference_deg);
    }

    #[test]
    fn vs_hold_commands_pitch_proportional_to_vs_error() {
        let mut ap = Autopilot::with_defaults();
        let s = sample(); // vs = 0
        ap.set_mode(ApMode::VsHold, &s);
        ap.set_vs_target(500.0);
        let out = ap.step(&s, 0.05);
        assert!(out.pitch_reference_deg > 0.0);
        // pitch command stays within structural limit
        assert!(out.pitch_reference_deg <= ap.config().max_pitch_deg + 1e-9);
    }

    #[test]
    fn roll_hold_tracks_bank_with_opposite_sign_error() {
        let mut ap = Autopilot::with_defaults();
        let mut s = sample();
        s.roll_deg = 0.0;
        ap.set_mode(ApMode::RollHold, &s);
        ap.set_roll_target(10.0);
        let out = ap.step(&s, 0.02);
        assert!(out.aileron > 0.0);
        assert!((out.roll_reference_deg - 10.0).abs() < 1e-9);
    }

    #[test]
    fn pitch_hold_direct_mode() {
        let mut ap = Autopilot::with_defaults();
        let s = sample();
        ap.set_mode(ApMode::PitchHold, &s);
        ap.set_pitch_target(5.0);
        let out = ap.step(&s, 0.02);
        assert!(out.elevator > 0.0);
        assert!((out.pitch_reference_deg - 5.0).abs() < 1e-9);
    }

    #[test]
    fn mode_switch_resets_integrators() {
        let mut ap = Autopilot::with_defaults();
        let s = sample();
        ap.set_mode(ApMode::AltitudeHold, &s);
        ap.set_altitude_target(9_000.0);
        for _ in 0..100 {
            ap.step(&s, 0.05);
        }
        // switching to RollHold clears the altitude loop's accumulated
        // integral: re-entering AltitudeHold starts from capture with
        // zero history (observable via a fresh first-step magnitude).
        ap.set_mode(ApMode::RollHold, &s);
        assert_eq!(ap.mode(), ApMode::RollHold);
        ap.set_mode(ApMode::AltitudeHold, &s);
        let s5 = ApSample { altitude_ft: 9_000.0, ..sample() };
        ap.set_altitude_target(9_000.0);
        let first = ap.step(&s5, 0.05);
        // at target with zero integral and derivative priming -> zero
        // elevator (nothing left to do)
        assert_eq!(first.elevator, 0.0, "elevator {}", first.elevator);
    }

    #[test]
    fn heading_target_normalized_to_0_360() {
        let mut ap = Autopilot::with_defaults();
        let s = sample();
        ap.set_mode(ApMode::RollHold, &s);
        ap.set_heading_target(-10.0);
        assert!((ap.heading_target_deg() - 350.0).abs() < 1e-9);
        ap.set_heading_target(720.0 + 33.0);
        assert!((ap.heading_target_deg() - 33.0).abs() < 1e-9);
    }

    #[test]
    fn wrap180_basics() {
        assert!((wrap180(0.0, 90.0) - 90.0).abs() < 1e-9);
        assert!((wrap180(0.0, 270.0) - -90.0).abs() < 1e-9);
        assert!((wrap180(350.0, 10.0) - 20.0).abs() < 1e-9);
        assert!((wrap180(10.0, 350.0) - -20.0).abs() < 1e-9);
        assert!((wrap180(180.0, -180.0)).abs() < 1e-9);
    }
}

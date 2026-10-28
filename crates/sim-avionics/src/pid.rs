//! Discrete PID controller with industrial safeguards.
//!
//! Features (all exercised by tests):
//! * **anti-windup** via conditional integration (the integral term only
//!   accumulates while the output is not saturated) plus an integrator
//!   clamp;
//! * **output saturation** to configured limits;
//! * **derivative-on-measurement** (no setpoint kick) with a first-order
//!   low-pass filter on the derivative term;
//! * safe handling of `dt <= 0`, non-finite inputs and repeated calls.

/// Controller gains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidGains {
    pub kp: f64,
    pub ki: f64,
    pub kd: f64,
}

impl PidGains {
    pub const fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd }
    }
}

/// Full controller configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidConfig {
    pub gains: PidGains,
    /// Lower output limit.
    pub out_min: f64,
    /// Upper output limit.
    pub out_max: f64,
    /// Time constant of the derivative low-pass filter in seconds;
    /// `0.0` disables filtering.
    pub deriv_filter_tau: f64,
    /// Use `d(error)/dt` (false, default) or `-d(measurement)/dt` (true).
    pub derivative_on_measurement: bool,
    /// Clamp applied to the integral term (`ki * integral` is the
    /// contribution before clamping is applied to the term itself).
    pub integrator_min: f64,
    pub integrator_max: f64,
}

impl PidConfig {
    pub fn new(gains: PidGains, out_min: f64, out_max: f64) -> Self {
        Self {
            gains,
            out_min,
            out_max,
            deriv_filter_tau: 0.0,
            derivative_on_measurement: false,
            integrator_min: f64::NEG_INFINITY,
            integrator_max: f64::INFINITY,
        }
    }

    pub fn with_derivative_filter(mut self, tau: f64) -> Self {
        self.deriv_filter_tau = tau;
        self
    }

    pub fn with_derivative_on_measurement(mut self) -> Self {
        self.derivative_on_measurement = true;
        self
    }

    pub fn with_integrator_limits(mut self, min: f64, max: f64) -> Self {
        self.integrator_min = min;
        self.integrator_max = max;
        self
    }
}

/// A single PID loop.
#[derive(Debug, Clone)]
pub struct Pid {
    config: PidConfig,
    /// Integral accumulator (already scaled by `ki`? no: raw `∫e dt`,
    /// the `ki` factor is applied at output assembly).
    integral: f64,
    /// Previous measurement (derivative-on-measurement) or error.
    prev_feedback: f64,
    /// Filtered derivative term.
    deriv_filtered: f64,
    /// Whether the first `step` has happened.
    primed: bool,
    /// Most recent saturated output (held across invalid inputs).
    last_out: f64,
}

impl Pid {
    pub fn new(config: PidConfig) -> Self {
        Self {
            config,
            integral: 0.0,
            prev_feedback: 0.0,
            deriv_filtered: 0.0,
            primed: false,
            last_out: 0.0,
        }
    }

    /// Convenience constructor: `Pid::new(PidConfig::new(gains, min, max))`.
    pub fn with_limits(gains: PidGains, out_min: f64, out_max: f64) -> Self {
        Self::new(PidConfig::new(gains, out_min, out_max))
    }

    pub fn config(&self) -> &PidConfig {
        &self.config
    }

    /// Replace the configuration, keeping state.
    pub fn set_config(&mut self, config: PidConfig) {
        self.config = config;
    }

    /// Current integral accumulator (`∫e dt`).
    pub fn integral(&self) -> f64 {
        self.integral
    }

    /// Reset all dynamic state.
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_feedback = 0.0;
        self.deriv_filtered = 0.0;
        self.primed = false;
        self.last_out = 0.0;
    }

    /// Advance one step. Returns the saturated controller output.
    ///
    /// * `setpoint` — desired value of the process variable,
    /// * `measurement` — current process variable,
    /// * `dt` — elapsed time in seconds (values `<= 0` return the last
    ///   output without changing state; non-finite inputs are ignored).
    pub fn step(&mut self, setpoint: f64, measurement: f64, dt: f64) -> f64 {
        if !setpoint.is_finite() || !measurement.is_finite() || !dt.is_finite() || dt <= 0.0 {
            // hold last state and re-report the previous output
            return self.last_out;
        }
        let error = setpoint - measurement;
        let g = self.config.gains;

        // --- derivative term --------------------------------------------
        let raw_deriv = if !self.primed {
            0.0
        } else if self.config.derivative_on_measurement {
            -(measurement - self.prev_feedback) / dt
        } else {
            (error - self.prev_feedback) / dt
        };
        if self.config.deriv_filter_tau > 0.0 {
            let alpha = dt / (self.config.deriv_filter_tau + dt);
            self.deriv_filtered += alpha * (raw_deriv - self.deriv_filtered);
        } else {
            self.deriv_filtered = raw_deriv;
        }

        // --- proportional + tentative integral --------------------------
        let p = g.kp * error;
        let candidate_integral = (self.integral + error * dt)
            .clamp(self.config.integrator_min, self.config.integrator_max);
        let d = g.kd * self.deriv_filtered;

        // Conditional integration: only accept the new integral when the
        // unsaturated output would not push further into saturation.
        let trial = p + g.ki * candidate_integral + d;
        let saturated = (trial > self.config.out_max && error > 0.0)
            || (trial < self.config.out_min && error < 0.0);
        if !saturated {
            self.integral = candidate_integral;
        }

        let raw = p + g.ki * self.integral + d;
        let out = raw.clamp(self.config.out_min, self.config.out_max);

        self.prev_feedback = if self.config.derivative_on_measurement {
            measurement
        } else {
            error
        };
        self.primed = true;
        self.last_out = out;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(kp: f64, ki: f64, kd: f64) -> PidConfig {
        PidConfig::new(PidGains::new(kp, ki, kd), -1.0, 1.0)
    }

    #[test]
    fn proportional_only_tracks_sign_and_magnitude() {
        let mut pid = Pid::new(cfg(0.5, 0.0, 0.0));
        let out = pid.step(1.0, 0.0, 0.01);
        assert!((out - 0.5).abs() < 1e-12); // 0.5 * 1.0
        let out2 = pid.step(1.0, 0.4, 0.01);
        assert!((out2 - 0.3).abs() < 1e-12); // 0.5 * 0.6
        let out3 = pid.step(-1.0, 0.0, 0.01);
        assert!(out3 < 0.0);
    }

    #[test]
    fn output_saturates_at_limits() {
        let mut pid = Pid::new(cfg(10.0, 0.0, 0.0));
        assert_eq!(pid.step(100.0, 0.0, 0.01), 1.0);
        assert_eq!(pid.step(-100.0, 0.0, 0.01), -1.0);
    }

    #[test]
    fn integral_accumulates_and_reduces_error() {
        // wide limits: this test watches the integrator, not saturation
        let mut pid = Pid::new(PidConfig::new(PidGains::new(0.0, 1.0, 0.0), -10.0, 10.0));
        let o1 = pid.step(1.0, 0.0, 0.5); // integral = 0.5
        assert!((o1 - 0.5).abs() < 1e-12);
        let o2 = pid.step(1.0, 0.0, 0.5); // integral = 1.0
        assert!((o2 - 1.0).abs() < 1e-12);
        // error shrinks but integral keeps its history
        let o3 = pid.step(1.0, 0.9, 0.5); // integral = 1.0 + 0.05
        assert!((o3 - 1.05).abs() < 1e-12);
        assert!((pid.integral() - 1.05).abs() < 1e-12);
    }

    #[test]
    fn anti_windup_limits_integral_when_saturated() {
        let mut pid = Pid::new(cfg(2.0, 5.0, 0.0));
        // Huge positive error: output saturates, integral must not grow.
        for _ in 0..50 {
            pid.step(1000.0, 0.0, 0.1);
        }
        assert!(
            pid.integral() == 0.0,
            "integral wound up: {}",
            pid.integral()
        );
    }

    #[test]
    fn integrator_clamp_bounds_state() {
        let mut pid = Pid::new(
            PidConfig::new(PidGains::new(0.0, 1.0, 0.0), -10.0, 10.0)
                .with_integrator_limits(-0.2, 0.2),
        );
        for _ in 0..100 {
            pid.step(1.0, 0.0, 0.5);
        }
        assert!((pid.integral() - 0.2).abs() < 1e-12);
        for _ in 0..1000 {
            pid.step(-1.0, 0.0, 0.5);
        }
        assert!((pid.integral() + 0.2).abs() < 1e-12);
    }

    #[test]
    fn derivative_on_measurement_has_no_setpoint_kick() {
        let mut pid = Pid::new(cfg(0.0, 0.0, 2.0).with_derivative_on_measurement());
        // First step primes; measurement steady.
        pid.step(0.0, 0.0, 0.1);
        // Setpoint jump with unchanged measurement: derivative term = 0.
        let out = pid.step(1000.0, 0.0, 0.1);
        assert_eq!(out, 0.0);

        // Measurement jump produces derivative response of opposite sign.
        let out2 = pid.step(1000.0, 5.0, 0.1);
        assert!(out2 < 0.0, "expected negative d term, got {out2}");
    }

    #[test]
    fn derivative_on_error_kicks_on_setpoint_jump() {
        let mut pid = Pid::new(cfg(0.0, 0.0, 2.0));
        pid.step(0.0, 0.0, 0.1);
        let out = pid.step(10.0, 0.0, 0.1); // error 0 -> 10
        assert!(out > 0.0, "expected positive kick, got {out}");
    }

    #[test]
    fn derivative_filter_smooths_spike() {
        let mut filtered = Pid::new(cfg(0.0, 0.0, 1.0).with_derivative_filter(1.0));
        let mut raw = Pid::new(cfg(0.0, 0.0, 1.0));
        filtered.step(0.0, 0.0, 0.05);
        raw.step(0.0, 0.0, 0.05);
        let of = filtered.step(0.0, 1.0, 0.05); // measurement jump
        let or_ = raw.step(0.0, 1.0, 0.05);
        assert!(of.abs() < or_.abs());
    }

    #[test]
    fn invalid_dt_and_inputs_hold_state() {
        let mut pid = Pid::new(cfg(1.0, 0.0, 0.0));
        let o1 = pid.step(2.0, 0.0, 0.01);
        let o2 = pid.step(2.0, 0.0, 0.0);
        assert_eq!(o2, o1);
        let o3 = pid.step(2.0, 0.0, -1.0);
        assert_eq!(o3, o1);
        let o4 = pid.step(f64::NAN, 0.0, 0.01);
        assert_eq!(o4, o1);
        assert_eq!(pid.integral(), 0.0);
    }

    #[test]
    fn reset_clears_everything() {
        // ki = 0 so the first post-reset step isolates the P term; the
        // integral still accumulates state for the reset check itself.
        let mut pid = Pid::new(PidConfig::new(PidGains::new(1.0, 0.0, 1.0), -100.0, 100.0));
        for _ in 0..10 {
            pid.step(5.0, 0.0, 0.1);
        }
        assert_ne!(pid.integral(), 0.0);
        pid.reset();
        assert_eq!(pid.integral(), 0.0);
        // after reset the first derivative contribution is zero:
        // out == p == 1.0 * 5.0, no kick from the stale history
        let out = pid.step(5.0, 0.0, 0.1);
        assert!((out - 5.0).abs() < 1e-9);
    }

    #[test]
    fn closes_on_constant_setpoint_p_only() {
        let mut pid = Pid::new(PidConfig::new(PidGains::new(0.8, 0.0, 0.0), -100.0, 100.0));
        let mut measurement = 0.0;
        for _ in 0..40 {
            let u = pid.step(50.0, measurement, 0.05);
            measurement += u; // plant: integrator 1/s
        }
        // With wide output limits the P-loop of a pure integrator is a
        // contraction: meas_{n+1} = meas_n + 0.8 (50 - meas_n) and the
        // fixed point is exactly the setpoint.
        assert!((measurement - 50.0).abs() < 1.0, "meas={measurement}");
    }
}

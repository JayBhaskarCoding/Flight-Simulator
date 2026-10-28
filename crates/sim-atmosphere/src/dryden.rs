//! Dryden discrete turbulence model (MIL-F-8785C public-domain form).
//!
//! The continuous Dryden spectra are realised as linear transfer-function
//! filters driven by band-limited white noise:
//!
//! * longitudinal `u'`: first-order lag with time constant `Tu = Lus/V`
//! * lateral/vertical `v'`, `w'`: second-order sections
//!   `H(s) = sigma·sqrt(2·Lv/(pi·V))·(1 + sqrt(3)·Lv/V·s) / (1 + Lv/V·s)^2`
//!
//! Continuous A/B (per axis, states `[x, x_dot]` for first order and
//! `[x, x_dot, x_ddot]` for second order) are discretised at the simulation
//! step with an exact matrix exponential (Van Loan), then the discrete noise
//! covariance is solved from the discrete Lyapunov equation
//! `C P Cᵀ = Sigma - P` with `P = expm([[A, BBᵀ],[0,-Aᵀ]])·[0;I]`, and the
//! noise gain is normalised as `g = sigma/sqrt((C P Cᵀ)₁₁)` so the stationary
//! variance of each component equals `sigma²` exactly.
//!
//! Angle perturbations are recovered from the gust velocities as
//! `delta_alpha ≈ -w'/V`, `delta_beta ≈ -v'/V`.

use glam::DVec3;

/// Turbulence intensity category with vertical-intensity reference (m/s).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurbulenceIntensity {
    /// `sigma_w = 1.7 m/s` (light chop).
    Light,
    /// `sigma_w = 3.7 m/s`.
    Moderate,
    /// `sigma_w = 7.5 m/s`.
    Severe,
}

impl TurbulenceIntensity {
    /// Vertical RMS intensity in m/s.
    pub fn sigma_w_mps(self) -> f64 {
        match self {
            Self::Light => 1.7,
            Self::Moderate => 3.7,
            Self::Severe => 7.5,
        }
    }

    /// Lateral RMS intensity (MIL ratio sigma_v/sigma_w = 0.7 at low level
    /// for the discrete-scale model used here).
    pub fn sigma_v_mps(self) -> f64 {
        self.sigma_w_mps() * 0.7
    }

    /// Longitudinal RMS intensity (sigma_u/sigma_w = 0.6 reference ratio).
    pub fn sigma_u_mps(self) -> f64 {
        self.sigma_w_mps() * 0.6
    }
}

/// Dryden length scales (m) at the current altitude for each axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrydenScales {
    /// Longitudinal scale `Lu`.
    pub longitudinal_m: f64,
    /// Lateral scale `Lv`.
    pub lateral_m: f64,
    /// Vertical scale `Lw`.
    pub vertical_m: f64,
}

impl DrydenScales {
    /// MIL-F-8785C low-altitude (`h <= 300 m`) closed form given turbulence
    /// intensity `sigma_w` (m/s) and altitude `h` (m, >= 0).
    pub fn low_altitude(_sigma_w_mps: f64, height_m: f64) -> Self {
        let h = height_m.max(0.0);
        Self {
            longitudinal_m: h / (0.177 + 0.000_149 * h).max(1e-6),
            lateral_m: h,
            vertical_m: h / (0.177 + 0.000_149 * h).max(1e-6),
        }
    }

    /// MIL-F-8785C high-altitude (`h > 300 m`) form.
    pub fn high_altitude(sigma_w_mps: f64, _height_m: f64) -> Self {
        let sigma = sigma_w_mps.max(1e-6);
        Self {
            longitudinal_m: 533.0 * sigma / 15.0,
            lateral_m: 533.0 * sigma / 15.0,
            vertical_m: 533.0 * sigma / 15.0,
        }
    }

    /// Combined selection: low-altitude form at and below 300 m,
    /// high-altitude form above (uses only `sigma_w` for consistency).
    pub fn for_conditions(sigma_w_mps: f64, height_m: f64) -> Self {
        if height_m <= 300.0 {
            Self::low_altitude(sigma_w_mps, height_m)
        } else {
            Self::high_altitude(sigma_w_mps, height_m)
        }
    }
}

/// Plain alias used by the crate re-exports: the in-crate generator.
pub type Rng = Xoshiro256;

/// Declarative configuration for a [`DrydenTurbulence`] instance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrydenConfig {
    /// True airspeed at the filter time constants, m/s.
    pub airspeed_mps: f64,
    /// Height above ground for the length-scale laws, m.
    pub height_m: f64,
    /// Turbulence intensity category.
    pub intensity: TurbulenceIntensity,
    /// Integration step, s (filters are exact for this step).
    pub step_s: f64,
    /// Deterministic seed.
    pub seed: u64,
}

impl DrydenConfig {
    /// Standard configuration shortcut.
    pub fn new(
        airspeed_mps: f64,
        height_m: f64,
        intensity: TurbulenceIntensity,
        step_s: f64,
        seed: u64,
    ) -> Self {
        Self { airspeed_mps, height_m, intensity, step_s, seed }
    }
}

// ---------------------------------------------------------------------------
// small fixed-size matrix helpers (f64, no external deps)
// ---------------------------------------------------------------------------

type M2 = [[f64; 2]; 2];

/// Exact `expm` of a 2x2 matrix via the closed-form hyperbolic method.
fn expm2(a: M2) -> M2 {
    let tr = a[0][0] + a[1][1];
    let s = tr / 2.0;
    let a_shift: M2 = [[a[0][0] - s, a[0][1]], [a[1][0], a[1][1] - s]];
    let dm = (a[0][0] - a[1][1]) / 2.0;
    let delta_sq = dm * dm + a[0][1] * a[1][0];
    let es = s.exp();
    if delta_sq >= 0.0 {
        let d = delta_sq.sqrt();
        let (ch, factor) = if d.abs() < 1e-12 {
            (1.0, 1.0)
        } else {
            ((d.exp() + (-d).exp()) / 2.0, (d.exp() - (-d).exp()) / (2.0 * d))
        };
        [
            [es * (ch + a_shift[0][0] * factor), es * (a_shift[0][1] * factor)],
            [es * (a_shift[1][0] * factor), es * (ch + a_shift[1][1] * factor)],
        ]
    } else {
        let d = (-delta_sq).sqrt();
        let (c, factor) = if d.abs() < 1e-12 {
            (1.0, 1.0)
        } else {
            (d.cos(), d.sin() / d)
        };
        [
            [es * (c + a_shift[0][0] * factor), es * (a_shift[0][1] * factor)],
            [es * (a_shift[1][0] * factor), es * (c + a_shift[1][1] * factor)],
        ]
    }
}

fn inv2(a: M2) -> Option<M2> {
    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if det.abs() < 1e-300 {
        return None;
    }
    let inv = 1.0 / det;
    Some([[a[1][1] * inv, -a[0][1] * inv], [-a[1][0] * inv, a[0][0] * inv]])
}

fn m2_mul(a: M2, b: M2) -> M2 {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

/// Solve the discrete Lyapunov equation `X = Phi X Phiᵀ + Q` by
/// fixed-point iteration (contractive for stable `Phi`).
fn discrete_lyapunov(phi: M2, q: M2) -> M2 {
    let mut x = q;
    for _ in 0..10_000 {
        let phi_x = m2_mul(phi, x);
        let phi_t = [[phi[0][0], phi[1][0]], [phi[0][1], phi[1][1]]];
        let next = m2_mul(phi_x, phi_t);
        let mut delta = 0.0f64;
        for i in 0..2 {
            for j in 0..2 {
                let n = next[i][j] + q[i][j];
                delta = delta.max((n - x[i][j]).abs());
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                x[i][j] = next[i][j] + q[i][j];
            }
        }
        if delta < 1e-14 {
            break;
        }
    }
    x
}

// ---------------------------------------------------------------------------
// first-order section (longitudinal u): Ornstein-Uhlenbeck AR(1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct FirstOrder {
    /// `x_{k+1} = phi x_k + b n_k`.
    phi: f64,
    /// Unit-variance noise gain `b` chosen so `var(x) = sigma^2`.
    noise_gain: f64,
}

impl FirstOrder {
    /// OU section with time constant `tau`, target RMS `sigma`, step `h`.
    fn new(tau: f64, sigma: f64, h: f64) -> Self {
        let tau = tau.max(1e-9);
        let phi = (-h / tau).exp();
        let b = (sigma * sigma * (1.0 - phi * phi)).sqrt();
        Self { phi, noise_gain: b }
    }
}

// ---------------------------------------------------------------------------
// second-order section (v, w): MIL-F-8785C shaping filter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct SecondOrder {
    /// Discrete state transition (2x2), states `[x, xdot]`.
    phi: M2,
    /// Normalised input column (2x1) for unit-variance noise.
    gamma: [f64; 2],
    /// Stationary state covariance under the normalised input.
    p: M2,
    /// Output row `[c0, c1]`: the gust velocity is `c0·x + c1·xdot`.
    c: [f64; 2],
}

impl SecondOrder {
    /// The turbulence (output) value for a state.
    fn output(&self, state: [f64; 2]) -> f64 {
        self.c[0] * state[0] + self.c[1] * state[1]
    }

    /// Stationary standard deviation of the output.
    fn output_sigma(&self) -> f64 {
        let c = self.c;
        let var = c[0] * (self.p[0][0] * c[0] + self.p[0][1] * c[1])
            + c[1] * (self.p[1][0] * c[0] + self.p[1][1] * c[1]);
        var.max(0.0).sqrt()
    }

    fn advance(&self, state: [f64; 2], noise: f64) -> [f64; 2] {
        [
            self.phi[0][0] * state[0] + self.phi[0][1] * state[1] + self.gamma[0] * noise,
            self.phi[1][0] * state[0] + self.phi[1][1] * state[1] + self.gamma[1] * noise,
        ]
    }
}

/// Build the MIL-F-8785C second-order transfer function
/// `H(s) = sigma*sqrt(2*L/(pi*V)) * (1 + sqrt(3)*L/V*s) / (1 + L/V*s)^2`
/// in controllable canonical form and discretise it exactly at step `h`.
fn second_order(sigma: f64, length_m: f64, velocity_mps: f64, h: f64) -> SecondOrder {
    let v = velocity_mps.max(1.0);
    let l = length_m.max(1.0);
    let tau = l / v;
    // H(s) = k (sqrt3 tau s + 1) / (tau^2 s^2 + 2 tau s + 1)
    let a0 = 1.0 / (tau * tau);
    let a1 = 2.0 / tau;
    let k = sigma * (2.0 * l / (std::f64::consts::PI * v)).sqrt();

    // controllable canonical form: x' = A x + B u, y = c·x
    let a: M2 = [[0.0, 1.0], [-a0, -a1]];
    let b = [0.0f64, 1.0f64];
    let c = [k, k * 3f64.sqrt() * tau];

    // exact ZOH: Phi = expm(A h);  Gamma = A^-1 (Phi - I) B
    let a_h = [[a[0][0] * h, a[0][1] * h], [a[1][0] * h, a[1][1] * h]];
    let phi = expm2(a_h);
    let a_inv = inv2(a).expect("canonical A is nonsingular (a0 > 0)");
    let phi_minus_i: M2 = [[phi[0][0] - 1.0, phi[0][1]], [phi[1][0], phi[1][1] - 1.0]];
    let pb = [
        phi_minus_i[0][0] * b[0] + phi_minus_i[0][1] * b[1],
        phi_minus_i[1][0] * b[0] + phi_minus_i[1][1] * b[1],
    ];
    let gamma = [
        a_inv[0][0] * pb[0] + a_inv[0][1] * pb[1],
        a_inv[1][0] * pb[0] + a_inv[1][1] * pb[1],
    ];

    // discrete stationary covariance: P = Phi P Phiᵀ + gamma gammaᵀ
    let q: M2 = [[gamma[0] * gamma[0], gamma[0] * gamma[1]], [gamma[1] * gamma[0], gamma[1] * gamma[1]]];
    let p_unit = discrete_lyapunov(phi, q);

    // scale the noise so the output variance equals sigma² exactly
    let mut so = SecondOrder { phi, gamma, p: p_unit, c };
    let os = so.output_sigma().max(1e-30);
    let scale = sigma / os;
    so.gamma = [gamma[0] * scale, gamma[1] * scale];
    so.p = [
        [p_unit[0][0] * scale * scale, p_unit[0][1] * scale * scale],
        [p_unit[1][0] * scale * scale, p_unit[1][1] * scale * scale],
    ];
    so
}

// ---------------------------------------------------------------------------
// xoshiro256++ PRNG (public-domain algorithm) + Box-Muller
// ---------------------------------------------------------------------------

/// Deterministic in-crate PRNG (xoshiro256++).
#[derive(Debug, Clone)]
pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed from a 64-bit value (SplitMix64 expansion).
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in (0, 1) (never exactly 0 or 1).
    pub fn next_f64(&mut self) -> f64 {
        // 53-bit uniform, offset to open interval
        let u = (self.next_u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (u + f64::EPSILON / 2.0).min(1.0 - f64::EPSILON / 2.0)
    }

    /// Standard normal deviate (Box-Muller, cached second value).
    pub fn next_gaussian(&mut self) -> f64 {
        // cache stored in s[0] as a marker is avoided; use simple loop
        // without caching to keep state explicit and testable.
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = std::f64::consts::TAU * u2;
                return r * theta.cos();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// the driven turbulence model
// ---------------------------------------------------------------------------

/// One simulation step of the Dryden turbulence state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurbulenceState {
    /// Body-axis gust velocities (u', v', w'), m/s.
    pub gust_mps: DVec3,
    /// Angle perturbation vector (delta alpha, delta beta, 0), radians.
    pub angle_perturbation_rad: DVec3,
}

/// Discrete Dryden turbulence generator.
#[derive(Debug, Clone)]
pub struct DrydenTurbulence {
    /// Simulated airspeed (m/s) — fixed for the section time constants.
    airspeed_mps: f64,
    /// Target RMS intensities per axis (m/s).
    sigma: DVec3,
    /// Length scales per axis (m).
    scales: DrydenScales,
    /// Longitudinal first-order section.
    u: FirstOrder,
    /// Lateral second-order section.
    v: SecondOrder,
    /// Vertical second-order section.
    w: SecondOrder,
    /// Section states: u is a scalar; v/w are `[x, xdot]` pairs.
    u_state: f64,
    v_state: [f64; 2],
    w_state: [f64; 2],
    /// RNG.
    rng: Xoshiro256,
}

impl DrydenTurbulence {
    /// Build a generator for the given conditions.
    ///
    /// * `airspeed_mps` — true airspeed used for time constants (m/s)
    /// * `height_m` — altitude used for the length-scale laws
    /// * `intensity` — turbulence category
    /// * `step_s` — simulation step (the generator is exact for this step)
    /// * `seed` — RNG seed
    pub fn new(
        airspeed_mps: f64,
        height_m: f64,
        intensity: TurbulenceIntensity,
        step_s: f64,
        seed: u64,
    ) -> Self {
        let sigma = DVec3::new(
            intensity.sigma_u_mps(),
            intensity.sigma_v_mps(),
            intensity.sigma_w_mps(),
        );
        let scales = DrydenScales::for_conditions(intensity.sigma_w_mps(), height_m);
        Self::with_scales(airspeed_mps, sigma, scales, step_s, seed)
    }

    /// Build from a [`DrydenConfig`].
    pub fn from_config(config: &DrydenConfig) -> Self {
        Self::new(
            config.airspeed_mps,
            config.height_m,
            config.intensity,
            config.step_s,
            config.seed,
        )
    }

    /// Build a generator with explicit RMS intensities and length scales.
    pub fn with_scales(
        airspeed_mps: f64,
        sigma: DVec3,
        scales: DrydenScales,
        step_s: f64,
        seed: u64,
    ) -> Self {
        let v = airspeed_mps.max(1.0);
        let tau_u = scales.longitudinal_m / v;
        Self {
            airspeed_mps: v,
            sigma,
            scales,
            u: FirstOrder::new(tau_u, sigma.x, step_s),
            v: second_order(sigma.y, scales.lateral_m, v, step_s),
            w: second_order(sigma.z, scales.vertical_m, v, step_s),
            u_state: 0.0,
            v_state: [0.0; 2],
            w_state: [0.0; 2],
            rng: Xoshiro256::seed_from_u64(seed),
        }
    }

    /// RMS intensities currently configured (m/s).
    pub fn sigma_mps(&self) -> DVec3 {
        self.sigma
    }

    /// Length scales currently configured (m).
    pub fn scales(&self) -> DrydenScales {
        self.scales
    }

    /// Advance one step and return the gust/angle state.
    pub fn step(&mut self) -> TurbulenceState {
        // first-order u
        let n_u = self.rng.next_gaussian();
        self.u_state = self.u.phi * self.u_state + self.u.noise_gain * n_u;
        let u = self.u_state;

        // second-order v, w
        let n_v = self.rng.next_gaussian();
        let n_w = self.rng.next_gaussian();
        self.v_state = self.v.advance(self.v_state, n_v);
        self.w_state = self.w.advance(self.w_state, n_w);
        let v_out = self.v.output(self.v_state);
        let w_out = self.w.output(self.w_state);

        let gust = DVec3::new(u, v_out, w_out);
        let v_airspeed = self.airspeed_mps;
        let angles = DVec3::new(
            -gust.z / v_airspeed, // delta alpha
            -gust.y / v_airspeed, // delta beta
            0.0,
        );
        TurbulenceState { gust_mps: gust, angle_perturbation_rad: angles }
    }

    /// Restart from zero state with a fresh seed.
    pub fn reseed(&mut self, seed: u64) {
        self.u_state = 0.0;
        self.v_state = [0.0; 2];
        self.w_state = [0.0; 2];
        self.rng = Xoshiro256::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variance(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0)
    }

    #[test]
    fn rng_is_deterministic_and_in_range() {
        let mut a = Xoshiro256::seed_from_u64(42);
        let mut b = Xoshiro256::seed_from_u64(42);
        for _ in 0..100 {
            let x = a.next_f64();
            assert_eq!(x, b.next_f64());
            assert!((0.0..1.0).contains(&x));
        }
        let mut c = Xoshiro256::seed_from_u64(7);
        let vals: Vec<f64> = (0..20_000).map(|_| c.next_gaussian()).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = variance(&vals);
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.08, "var {var}");
    }

    #[test]
    fn intensity_levels() {
        assert_eq!(TurbulenceIntensity::Light.sigma_w_mps(), 1.7);
        assert_eq!(TurbulenceIntensity::Moderate.sigma_w_mps(), 3.7);
        assert_eq!(TurbulenceIntensity::Severe.sigma_w_mps(), 7.5);
        assert!(TurbulenceIntensity::Severe.sigma_u_mps() > TurbulenceIntensity::Light.sigma_u_mps());
    }

    #[test]
    fn length_scales_follow_mil_laws() {
        let low = DrydenScales::for_conditions(3.7, 150.0);
        assert!((low.lateral_m - 150.0).abs() < 1e-9);
        assert!(low.longitudinal_m > low.lateral_m);
        let high = DrydenScales::for_conditions(3.7, 2000.0);
        let expected = 533.0 * 3.7 / 15.0;
        assert!((high.longitudinal_m - expected).abs() < 1e-9);
    }

    #[test]
    fn first_order_variance_matches_target() {
        // stationary variance of the OU section must equal sigma^2
        let sigma = 2.5;
        let sec = FirstOrder::new(5.0, sigma, 0.02);
        let var = sec.noise_gain * sec.noise_gain / (1.0 - sec.phi * sec.phi);
        assert!((var.sqrt() - sigma).abs() < 1e-9, "var {}", var.sqrt());
    }

    #[test]
    fn second_order_output_variance_matches_target() {
        let sigma = 3.0;
        let sec = second_order(sigma, 300.0, 60.0, 0.01);
        let out_sigma = sec.output_sigma();
        assert!(
            (out_sigma - sigma).abs() < 1e-6 * sigma,
            "output sigma {out_sigma} vs {sigma}"
        );
    }

    #[test]
    fn expm2_closed_form_cases() {
        // identity
        let e = expm2([[0.0, 0.0], [0.0, 0.0]]);
        assert!((e[0][0] - 1.0).abs() < 1e-12);
        assert!(e[0][1].abs() < 1e-14);
        // diagonal
        let d = expm2([[-1.0, 0.0], [0.0, -2.0]]);
        assert!((d[0][0] - (-1.0f64).exp()).abs() < 1e-10);
        assert!((d[1][1] - (-2.0f64).exp()).abs() < 1e-10);
        // rotation-like nilpotent shift: exp([[0,1],[0,0]]) = [[1,1],[0,1]]
        let n = expm2([[0.0, 1.0], [0.0, 0.0]]);
        assert!((n[0][0] - 1.0).abs() < 1e-12);
        assert!((n[0][1] - 1.0).abs() < 1e-12);
        assert!(n[1][0].abs() < 1e-14);
        // complex-eigenvalue case: pure rotation [[0,-w],[w,0]]
        let w = 2.0f64;
        let r = expm2([[0.0, -w], [w, 0.0]]);
        assert!((r[0][0] - w.cos()).abs() < 1e-10);
        assert!((r[0][1] - (-w.sin())).abs() < 1e-10);
    }

    #[test]
    fn discrete_lyapunov_diagonal_case() {
        // X = Phi X Phiᵀ + Q with diagonal phi: X_ii = Q_ii / (1 - phi_ii²)
        let phi: M2 = [[0.9, 0.0], [0.0, 0.5]];
        let q: M2 = [[1.0, 0.0], [0.0, 2.0]];
        let x = discrete_lyapunov(phi, q);
        assert!((x[0][0] - 1.0 / (1.0 - 0.81)).abs() < 1e-9, "{}", x[0][0]);
        assert!((x[1][1] - 2.0 / (1.0 - 0.25)).abs() < 1e-9, "{}", x[1][1]);
        assert!(x[0][1].abs() < 1e-12);
    }

    #[test]
    fn turbulence_statistics_match_targets() {
        let step = 0.01;
        let mut t = DrydenTurbulence::new(60.0, 300.0, TurbulenceIntensity::Moderate, step, 1234);
        let n = 200_000;
        let mut u_samples = Vec::with_capacity(n);
        let mut w_samples = Vec::with_capacity(n);
        for _ in 0..n {
            let s = t.step();
            u_samples.push(s.gust_mps.x);
            w_samples.push(s.gust_mps.z);
        }
        let var_u = variance(&u_samples).sqrt();
        let var_w = variance(&w_samples).sqrt();
        let target_u = 3.7 * 0.6;
        assert!(
            (var_u - target_u).abs() / target_u < 0.15,
            "sigma_u {var_u} vs {target_u}"
        );
        assert!(
            (var_w - 3.7).abs() / 3.7 < 0.12,
            "sigma_w {var_w} vs 3.7"
        );
    }

    #[test]
    fn angle_perturbations_scale_with_gusts() {
        let mut t = DrydenTurbulence::new(50.0, 500.0, TurbulenceIntensity::Severe, 0.02, 99);
        let mut max_alpha = 0.0f64;
        for _ in 0..20_000 {
            let s = t.step();
            let expected = -s.gust_mps.z / 50.0;
            assert!((s.angle_perturbation_rad.x - expected).abs() < 1e-15);
            max_alpha = max_alpha.max(s.angle_perturbation_rad.x.abs());
        }
        // 7.5 m/s sigma at 50 m/s → alpha RMS roughly 0.15 rad scale
        assert!(max_alpha > 0.05, "max alpha {max_alpha}");
    }

    #[test]
    fn reseed_restarts_identically() {
        let mut a = DrydenTurbulence::new(60.0, 300.0, TurbulenceIntensity::Light, 0.01, 5);
        let first: Vec<DVec3> = (0..10).map(|_| a.step().gust_mps).collect();
        a.reseed(5);
        for (i, expect) in first.iter().enumerate() {
            let got = a.step().gust_mps;
            assert!((*expect - got).length() < 1e-12, "step {i}");
        }
    }

    #[test]
    fn zero_turbulence_is_quiet() {
        let scales = DrydenScales { longitudinal_m: 500.0, lateral_m: 500.0, vertical_m: 500.0 };
        let sigma = DVec3::ZERO;
        let mut t = DrydenTurbulence::with_scales(60.0, sigma, scales, 0.01, 1);
        for _ in 0..1000 {
            let s = t.step();
            assert!(s.gust_mps.length() < 1e-9);
            assert!(s.angle_perturbation_rad.length() < 1e-12);
        }
    }
}

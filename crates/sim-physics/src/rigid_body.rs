//! Massive rigid body with an explicit RK4 integrator (f64 throughout).

use glam::{DMat3, DQuat, DVec3, DVec4};

/// A force/moment pair applied to the body for one evaluation.
///
/// The force is expressed in the **world** frame and the torque in the
/// **body** frame (the natural frames for Newton and Euler equations).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ForceTorque {
    /// Net force on the CG, world frame, N.
    pub force_world: DVec3,
    /// Net moment about the CG, body frame, N·m.
    pub torque_body: DVec3,
}

impl ForceTorque {
    /// The zero wrench.
    pub const ZERO: Self = Self { force_world: DVec3::ZERO, torque_body: DVec3::ZERO };
}

impl std::ops::Add for ForceTorque {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            force_world: self.force_world + other.force_world,
            torque_body: self.torque_body + other.torque_body,
        }
    }
}

/// Full dynamic state of the rigid body.
///
/// Positions/velocities are world-frame (ENU), the angular rate is in the
/// body frame, and `orientation` maps body → world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBodyState {
    /// Centre-of-mass position, world frame, m.
    pub position_m: DVec3,
    /// Body → world attitude.
    pub orientation: DQuat,
    /// Centre-of-mass velocity, world frame, m/s.
    pub velocity_mps: DVec3,
    /// Angular velocity about the body axes, rad/s.
    pub omega_body_radps: DVec3,
}

impl Default for RigidBodyState {
    fn default() -> Self {
        Self {
            position_m: DVec3::ZERO,
            orientation: DQuat::IDENTITY,
            velocity_mps: DVec3::ZERO,
            omega_body_radps: DVec3::ZERO,
        }
    }
}

impl RigidBodyState {
    /// Upright level attitude for an aeronautical `heading_rad`
    /// (clockwise from north): body `+X` points along the heading,
    /// body `+Y` to the right (horizontal), body `+Z` **down** (matching
    /// world-down in the ENU frame).
    ///
    /// Built as yaw about world `+Z` after a 180° roll that maps body-down
    /// onto world-down — a yaw-only quaternion would leave the Z-down body
    /// inverted in the ENU world.
    pub fn upright_attitude(heading_rad: f64) -> DQuat {
        DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2 - heading_rad)
            * DQuat::from_rotation_x(std::f64::consts::PI)
    }

    /// Level flight at `altitude_m` heading `heading_rad` clockwise from
    /// north (aeronautical heading), moving at `airspeed_mps`.
    pub fn level_flight(altitude_m: f64, heading_rad: f64, airspeed_mps: f64) -> Self {
        let orientation = Self::upright_attitude(heading_rad);
        let velocity_mps = orientation * DVec3::X * airspeed_mps;
        Self {
            position_m: DVec3::new(0.0, 0.0, altitude_m),
            orientation,
            velocity_mps,
            omega_body_radps: DVec3::ZERO,
        }
    }

    /// Rotate a body-frame vector into the world frame.
    pub fn body_to_world(&self, v: DVec3) -> DVec3 {
        self.orientation * v
    }

    /// Rotate a world-frame vector into the body frame.
    pub fn world_to_body(&self, v: DVec3) -> DVec3 {
        self.orientation.conjugate() * v
    }

    /// World-frame linear momentum.
    pub fn linear_momentum(&self, mass_kg: f64) -> DVec3 {
        self.velocity_mps * mass_kg
    }

    /// World-frame angular momentum `R · (I ω)`.
    pub fn angular_momentum_world(&self, inertia_body: DMat3) -> DVec3 {
        self.orientation * (inertia_body * self.omega_body_radps)
    }

    /// Translational + rotational kinetic energy, J.
    pub fn kinetic_energy(&self, mass_kg: f64, inertia_body: DMat3) -> f64 {
        let v = 0.5 * mass_kg * self.velocity_mps.length_squared();
        let w = self.omega_body_radps;
        let r = 0.5 * w.dot(inertia_body * w);
        v + r
    }

    /// True when every component is finite.
    pub fn is_finite(&self) -> bool {
        self.position_m.is_finite()
            && self.velocity_mps.is_finite()
            && self.omega_body_radps.is_finite()
            && self.orientation.is_finite()
            && (self.orientation.length() - 1.0).abs() < 1e-6
    }

    /// Normalise the attitude quaternion (constraint projection).
    pub fn normalized(&self) -> Self {
        let mut s = *self;
        s.orientation = s.orientation.normalize();
        s
    }

    /// `s + h · d` for one RK stage (no normalisation).
    fn combine(&self, d: &StateDeriv, h: f64) -> Self {
        Self {
            position_m: self.position_m + d.d_position * h,
            orientation: DQuat::from_xyzw(
                self.orientation.x + d.d_orientation.x * h,
                self.orientation.y + d.d_orientation.y * h,
                self.orientation.z + d.d_orientation.z * h,
                self.orientation.w + d.d_orientation.w * h,
            ),
            velocity_mps: self.velocity_mps + d.d_velocity * h,
            omega_body_radps: self.omega_body_radps + d.d_omega * h,
        }
    }
}

/// Time derivative of a [`RigidBodyState`] (quaternion derivative kept in
/// all four components — dropping `w` would break the kinematics).
#[derive(Debug, Clone, Copy, Default)]
struct StateDeriv {
    d_position: DVec3,
    d_orientation: DVec4,
    d_velocity: DVec3,
    d_omega: DVec3,
}

impl StateDeriv {
    /// `d · h` scaled for RK stage combination.
    fn scale(&self, h: f64) -> Self {
        Self {
            d_position: self.d_position * h,
            d_orientation: self.d_orientation * h,
            d_velocity: self.d_velocity * h,
            d_omega: self.d_omega * h,
        }
    }

    fn add(&self, o: &Self) -> Self {
        Self {
            d_position: self.d_position + o.d_position,
            d_orientation: self.d_orientation + o.d_orientation,
            d_velocity: self.d_velocity + o.d_velocity,
            d_omega: self.d_omega + o.d_omega,
        }
    }
}

/// Error returned when an inertia tensor is not physically valid.
#[derive(Debug, Clone, PartialEq)]
pub enum InertiaError {
    /// Matrix is not symmetric within `1e-9` relative tolerance.
    Asymmetric,
    /// Cholesky-style leading-principal-minor test failed (not positive
    /// definite) — e.g. a triangle inequality violation.
    NotPositiveDefinite,
}

impl std::fmt::Display for InertiaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Asymmetric => write!(f, "inertia tensor is not symmetric"),
            Self::NotPositiveDefinite => write!(f, "inertia tensor is not positive definite"),
        }
    }
}

impl std::error::Error for InertiaError {}

/// Mass properties of the rigid body (constant during a run; fuel mass
/// changes are re-initialised by the caller between steps if required).
#[derive(Debug, Clone, Copy)]
pub struct RigidBody {
    /// Mass, kg.
    pub mass_kg: f64,
    /// Inertia tensor about the CG in body axes, kg·m².
    pub inertia_body: DMat3,
    /// Inverse of [`Self::inertia_body`].
    pub inertia_body_inv: DMat3,
}

impl RigidBody {
    /// Body with a diagonal (principal-axis) inertia.
    pub fn from_diagonal(mass_kg: f64, principal: DVec3) -> Self {
        assert!(mass_kg > 0.0, "mass must be positive");
        assert!(
            principal.min_element() > 0.0,
            "principal inertias must be positive"
        );
        let inertia_body = DMat3::from_diagonal(principal);
        Self {
            mass_kg,
            inertia_body,
            inertia_body_inv: DMat3::from_diagonal(DVec3::ONE / principal),
        }
    }

    /// Body with a full symmetric positive-definite inertia (validated).
    pub fn try_from_mat3(mass_kg: f64, inertia_body: DMat3) -> Result<Self, InertiaError> {
        if mass_kg <= 0.0 {
            return Err(InertiaError::NotPositiveDefinite);
        }
        let m = inertia_body;
        let scale = m.x_axis.length().max(m.y_axis.length()).max(m.z_axis.length());
        let tol = 1e-9 * scale.max(1.0);
        for (a, b) in [
            (m.x_axis.y, m.y_axis.x),
            (m.x_axis.z, m.z_axis.x),
            (m.y_axis.z, m.z_axis.y),
        ] {
            if (a - b).abs() > tol {
                return Err(InertiaError::Asymmetric);
            }
        }
        // symmetric LDLᵀ without pivoting: succeeds iff SPD for symmetric input
        let mut l = DMat3::ZERO;
        let mut d = [0.0f64; 3];
        let cols = [m.x_axis, m.y_axis, m.z_axis];
        for j in 0..3 {
            let mut diag = cols[j][j];
            for (k, dk) in d[..j].iter().enumerate() {
                diag -= l.col(k)[j] * l.col(k)[j] * dk;
            }
            if diag <= 0.0 {
                return Err(InertiaError::NotPositiveDefinite);
            }
            d[j] = diag;
            l.col_mut(j)[j] = 1.0;
            for (offset, col_i) in cols[j + 1..].iter().enumerate() {
                let i = j + 1 + offset;
                let mut s = col_i[j];
                for (k, dk) in d[..j].iter().enumerate() {
                    s -= l.col(k)[i] * l.col(k)[j] * dk;
                }
                l.col_mut(j)[i] = s / diag;
            }
        }
        // L is unit-lower-triangular in column form; invert via forward
        // substitution applied to the identity to obtain L^-1, then
        // inertia^-1 = L^-T D^-1 L^-1.
        let linv = invert_unit_lower(l)?;
        let dinv = DMat3::from_diagonal(DVec3::new(1.0 / d[0], 1.0 / d[1], 1.0 / d[2]));
        let inertia_body_inv = linv.transpose() * dinv * linv;
        Ok(Self { mass_kg, inertia_body, inertia_body_inv })
    }

    /// Newton-Euler derivative for `(state, force_world, torque_body)`.
    pub fn derivative(
        &self,
        state: &RigidBodyState,
        forces: ForceTorque,
    ) -> (DVec3, DQuat, DVec3, DVec3) {
        let d_position = state.velocity_mps;
        // body rates: q̇ = ½ q ⊗ ω_body
        let d_orientation =
            (state.orientation * DQuat::from_xyzw(
                state.omega_body_radps.x,
                state.omega_body_radps.y,
                state.omega_body_radps.z,
                0.0,
            )) * 0.5;
        let d_velocity = forces.force_world / self.mass_kg;
        let i_omega = self.inertia_body * state.omega_body_radps;
        let gyro = state.omega_body_radps.cross(i_omega);
        let d_omega = self.inertia_body_inv * (forces.torque_body - gyro);
        (d_position, d_orientation, d_velocity, d_omega)
    }

    /// One fixed-step RK4 advance; `forces` is re-evaluated at all four
    /// stages (zero-order-hold on control inputs and environment).
    ///
    /// The attitude quaternion is projected to unit length after every
    /// stage composition and at the end of the step.
    pub fn rk4_step(
        &self,
        state: &RigidBodyState,
        h: f64,
        forces: impl Fn(&RigidBodyState) -> ForceTorque,
    ) -> RigidBodyState {
        let (p1, q1, v1, w1) = self.derivative(state, forces(state));
        let k1 = StateDeriv { d_position: p1, d_orientation: quat_vec4(q1), d_velocity: v1, d_omega: w1 };

        let s2 = state.combine(&k1, 0.5 * h).normalized();
        let (p2, q2, v2, w2) = self.derivative(&s2, forces(&s2));
        let k2 = StateDeriv { d_position: p2, d_orientation: quat_vec4(q2), d_velocity: v2, d_omega: w2 };

        let s3 = state.combine(&k2, 0.5 * h).normalized();
        let (p3, q3, v3, w3) = self.derivative(&s3, forces(&s3));
        let k3 = StateDeriv { d_position: p3, d_orientation: quat_vec4(q3), d_velocity: v3, d_omega: w3 };

        let s4 = state.combine(&k3, h).normalized();
        let (p4, q4, v4, w4) = self.derivative(&s4, forces(&s4));
        let k4 = StateDeriv { d_position: p4, d_orientation: quat_vec4(q4), d_velocity: v4, d_omega: w4 };

        // s' = s + (h/6)(k1 + 2k2 + 2k3 + k4)
        let sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
        let avg = sum.scale(1.0 / 6.0);
        state.combine(&avg, h).normalized()
    }
}

fn quat_vec4(q: DQuat) -> DVec4 {
    DVec4::new(q.x, q.y, q.z, q.w)
}

fn invert_unit_lower(l: DMat3) -> Result<DMat3, InertiaError> {
    // Solve L X = I column by column (forward substitution).
    let mut x = DMat3::IDENTITY;
    for col in 0..3 {
        let mut c = DMat3::IDENTITY.col(col);
        for i in 0..3 {
            for j in 0..i {
                c[i] -= l.col(j)[i] * c[j];
            }
            // unit diagonal
            if l.col(i)[i] != 1.0 {
                return Err(InertiaError::NotPositiveDefinite);
            }
        }
        *x.col_mut(col) = c;
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_forces(_: &RigidBodyState) -> ForceTorque {
        ForceTorque::ZERO
    }

    #[test]
    fn constant_velocity_is_exact() {
        let body = RigidBody::from_diagonal(100.0, DVec3::new(10.0, 20.0, 30.0));
        let s0 = RigidBodyState {
            position_m: DVec3::new(1.0, 2.0, 3.0),
            velocity_mps: DVec3::new(10.0, -5.0, 2.5),
            ..Default::default()
        };
        let mut s = s0;
        for _ in 0..1000 {
            s = body.rk4_step(&s, 0.001, zero_forces);
        }
        let expected = s0.position_m + s0.velocity_mps * 1.0;
        assert!((s.position_m - expected).length() < 1e-12);
        assert_eq!(s.velocity_mps, s0.velocity_mps);
        assert!((s.orientation.length() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn ballistic_arc_matches_analytic() {
        let body = RigidBody::from_diagonal(10.0, DVec3::ONE);
        let s0 = RigidBodyState {
            position_m: DVec3::ZERO,
            velocity_mps: DVec3::new(30.0, 0.0, 40.0),
            ..Default::default()
        };
        let g = DVec3::new(0.0, 0.0, -crate::G0);
        let gravity_force = g * body.mass_kg;
        let forces = |_s: &RigidBodyState| ForceTorque {
            force_world: gravity_force,
            torque_body: DVec3::ZERO,
        };
        let mut s = s0;
        for _ in 0..2000 {
            s = body.rk4_step(&s, 0.001, forces);
        }
        let t = 2.0f64;
        let expected = s0.position_m + s0.velocity_mps * t + 0.5 * g * t * t;
        let err = (s.position_m - expected).length();
        assert!(err < 1e-6, "RK4 ballistic error {err} m");
    }

    #[test]
    fn quaternion_stays_unit_under_tumble() {
        let body = RigidBody::from_diagonal(50.0, DVec3::new(5.0, 8.0, 12.0));
        let mut s = RigidBodyState {
            omega_body_radps: DVec3::new(3.0, -2.0, 1.5),
            velocity_mps: DVec3::new(1.0, 2.0, 3.0),
            ..Default::default()
        };
        for _ in 0..100_000 {
            s = body.rk4_step(&s, 0.001, zero_forces);
            assert!(s.is_finite());
        }
        assert!((s.orientation.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn torque_free_angular_momentum_conserved() {
        let principal = DVec3::new(5.0, 8.0, 12.0);
        let body = RigidBody::from_diagonal(50.0, principal);
        let inertia = DMat3::from_diagonal(principal);
        let mut s = RigidBodyState {
            omega_body_radps: DVec3::new(2.0, 1.0, 0.5),
            ..Default::default()
        };
        let l0 = s.angular_momentum_world(inertia);
        let e0 = s.kinetic_energy(body.mass_kg, inertia);
        for _ in 0..20_000 {
            s = body.rk4_step(&s, 0.001, zero_forces);
        }
        let l1 = s.angular_momentum_world(inertia);
        let e1 = s.kinetic_energy(body.mass_kg, inertia);
        assert!((l1 - l0).length() < 1e-6 * l0.length(), "ΔL = {}", (l1 - l0).length());
        assert!((e1 - e0).abs() < 1e-6 * e0, "ΔE = {}", e1 - e0);
    }

    #[test]
    fn principal_spin_rate_is_steady() {
        // pure spin about a principal axis: ω must not change
        let body = RigidBody::from_diagonal(10.0, DVec3::new(4.0, 9.0, 16.0));
        let mut s = RigidBodyState {
            omega_body_radps: DVec3::new(0.0, 5.0, 0.0),
            ..Default::default()
        };
        for _ in 0..10_000 {
            s = body.rk4_step(&s, 0.001, zero_forces);
        }
        assert!((s.omega_body_radps - DVec3::new(0.0, 5.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn quaternion_multiplication_sign_pitch_up() {
        // body +Y rate (nose up in the Z-down body frame) with identity
        // start: after time t the forward axis must pitch up by ωt.
        let body = RigidBody::from_diagonal(10.0, DVec3::ONE);
        let omega = 0.4f64;
        let mut s = RigidBodyState {
            omega_body_radps: DVec3::new(0.0, omega, 0.0),
            ..Default::default()
        };
        for _ in 0..5000 {
            s = body.rk4_step(&s, 0.001, zero_forces);
        }
        let fwd = s.body_to_world(DVec3::X);
        // after 5 s the body has pitched up by omega*t about +Y:
        // forward tilts toward −Z (world up from identity)
        let angle = omega * 5.0;
        let expected = DVec3::new(angle.cos(), 0.0, -angle.sin());
        assert!((fwd - expected).length() < 1e-9, "fwd = {fwd}, expected = {expected}");
    }

    #[test]
    fn inertia_validation() {
        assert!(RigidBody::try_from_mat3(1.0, DMat3::from_diagonal(DVec3::ONE)).is_ok());
        let asym = DMat3::from_cols_array(&[1.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(
            matches!(
                RigidBody::try_from_mat3(1.0, asym),
                Err(InertiaError::Asymmetric)
            )
        );
        // positive definite but violating the triangle inequality is still
        // caught when a leading minor goes non-positive; a plain indefinite
        // matrix must fail:
        let indef = DMat3::from_cols_array(&[1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        assert!(RigidBody::try_from_mat3(1.0, indef).is_err());
        assert!(RigidBody::try_from_mat3(-1.0, DMat3::IDENTITY).is_err());
    }

    #[test]
    fn full_inertia_inverse_is_consistent() {
        // symmetric SPD tensor: I · I^-1 = identity
        let i = DMat3::from_cols_array(&[20.0, 3.0, 1.0, 3.0, 15.0, 2.0, 1.0, 2.0, 10.0]);
        let body = RigidBody::try_from_mat3(5.0, i).unwrap();
        let prod = i * body.inertia_body_inv;
        let id = DMat3::IDENTITY;
        for c in 0..3 {
            assert!((prod.col(c) - id.col(c)).length() < 1e-10, "col {c}");
        }
    }

    #[test]
    fn constant_torque_angular_acceleration() {
        let body = RigidBody::from_diagonal(1.0, DVec3::new(2.0, 4.0, 8.0));
        let tau = DVec3::new(0.0, 6.0, 0.0); // about +Y → α = 6/4 = 1.5
        let s0 = RigidBodyState::default();
        let forces = |_s: &RigidBodyState| ForceTorque {
            force_world: DVec3::ZERO,
            torque_body: tau,
        };
        let mut s = s0;
        for _ in 0..1000 {
            s = body.rk4_step(&s, 0.001, forces);
        }
        let expected_w = 1.5f64; // rad/s after 1 s
        assert!((s.omega_body_radps.y - expected_w).abs() < 1e-9);
        // orientation = rotation by ½ α t² = 0.75 rad about Y
        let angle = 2.0 * (s.orientation.w).acos();
        assert!((angle - 0.75).abs() < 1e-6, "angle {angle}");
    }

    #[test]
    fn level_flight_heading_east() {
        let s = RigidBodyState::level_flight(300.0, std::f64::consts::FRAC_PI_2, 60.0);
        assert!((s.position_m.z - 300.0).abs() < 1e-12);
        let v = s.velocity_mps;
        assert!((v - DVec3::new(60.0, 0.0, 0.0)).length() < 1e-9, "v = {v}");
    }
}

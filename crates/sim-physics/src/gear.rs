//! Landing gear: penalty-contact struts with steered, friction-limited
//! tyres over a pluggable ground model.

use glam::{DQuat, DVec3};

use crate::{ControlInputs, RigidBodyState};

/// Ground surface sample at one planar location.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroundSurface {
    /// Surface height along world `+Z` at the queried point, m.
    pub height_m: f64,
    /// Unit surface normal (points "up", world frame).
    pub normal: DVec3,
}

/// Terrain query used by the gear solver.
///
/// The physics crate only needs height + normal; `sim-world` will provide
/// a DEM/ellipsoid-backed implementation later.
pub trait GroundProvider: Send + Sync {
    /// Surface height and normal at world `(x, y)`.
    fn surface_at(&self, x: f64, y: f64) -> GroundSurface;
}

/// Flat plane at a constant altitude with a vertical normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneGround {
    /// Height of the plane, m (world Z).
    pub height_m: f64,
}

impl PlaneGround {
    /// Plane at `height_m`.
    pub fn new(height_m: f64) -> Self {
        Self { height_m }
    }
}

impl GroundProvider for PlaneGround {
    fn surface_at(&self, _x: f64, _y: f64) -> GroundSurface {
        GroundSurface { height_m: self.height_m, normal: DVec3::Z }
    }
}

/// Static configuration of one gear leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearLegConfig {
    /// Strut top attachment, body frame (m, `+Z` down).
    pub attach_body: DVec3,
    /// Distance from the attachment to the wheel contact patch at full
    /// extension, m (along the body down direction).
    pub strut_length_m: f64,
    /// Vertical spring rate, N/m.
    pub spring_n_per_m: f64,
    /// Vertical damping, N·s/m.
    pub damper_n_s_per_m: f64,
    /// Maximum compression before the mechanical stop, m.
    pub max_compression_m: f64,
    /// Wheel radius, m (used for the bottoming guard).
    pub wheel_radius_m: f64,
    /// `true` when the leg steers with [`ControlInputs::steering`].
    pub steerable: bool,
    /// Maximum steering angle at full command, rad (positive = right).
    pub max_steer_rad: f64,
    /// `true` when the leg brakes with [`ControlInputs::brake`].
    pub braked: bool,
    /// Rolling-resistance coefficient (dimensionless, times normal load).
    pub rolling_resistance: f64,
    /// Longitudinal friction coefficient.
    pub mu_long: f64,
    /// Lateral friction coefficient.
    pub mu_lat: f64,
    /// Cornering stiffness of the tyre, N per rad of slip angle.
    pub cornering_stiffness: f64,
}

/// Which controls actuate an axle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GearAxle {
    /// Steerable nose wheel without brakes.
    Nose,
    /// Braked, non-steerable main wheel.
    Main,
    /// Fully custom actuation.
    Custom { steerable: bool, braked: bool },
}

impl GearLegConfig {
    /// A conventional clean-room tricycle-gear leg.
    ///
    /// `attach_body` + `strut_length_m` define the fully extended contact
    /// patch; the patch sits at `attach_body + (0, 0, strut_length_m)`
    /// (body `+Z` is down).
    pub fn tricycle(
        attach_body: DVec3,
        strut_length_m: f64,
        spring_n_per_m: f64,
        damper_n_s_per_m: f64,
        max_compression_m: f64,
        wheel_radius_m: f64,
        axle: GearAxle,
    ) -> Self {
        let (steerable, braked) = match axle {
            GearAxle::Nose => (true, false),
            GearAxle::Main => (false, true),
            GearAxle::Custom { steerable, braked } => (steerable, braked),
        };
        Self {
            attach_body,
            strut_length_m,
            spring_n_per_m,
            damper_n_s_per_m,
            max_compression_m,
            wheel_radius_m,
            steerable,
            max_steer_rad: 25.0f64.to_radians(),
            braked,
            rolling_resistance: 0.02,
            mu_long: 0.8,
            mu_lat: 0.9,
            cornering_stiffness: 30_000.0,
        }
    }
}

/// Runtime state of one gear leg (observability + contact bookkeeping).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearLeg {
    /// Static configuration.
    pub config: GearLegConfig,
    /// Current vertical compression, m.
    pub compression_m: f64,
    /// Normal load during the last solve, N.
    pub normal_load_n: f64,
    /// `true` while the wheel touches the ground.
    pub in_contact: bool,
}

impl GearLeg {
    /// New leg with the wheel fully extended.
    pub fn new(config: GearLegConfig) -> Self {
        Self { config, compression_m: 0.0, normal_load_n: 0.0, in_contact: false }
    }

    /// Fully extended world contact patch of this leg for `state`.
    pub fn contact_point_world(&self, state: &RigidBodyState) -> DVec3 {
        let patch_body =
            self.config.attach_body + DVec3::new(0.0, 0.0, self.config.strut_length_m);
        state.position_m + state.orientation * patch_body
    }
}

/// Contact wrench produced by one leg during a solve.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GearContact {
    /// Total contact force, world frame, N (zero when airborne).
    pub force_world: DVec3,
    /// Moment about the CG, body frame, N·m.
    pub torque_body: DVec3,
    /// `true` when compression exceeded the design travel (bottoming).
    pub bottomed_out: bool,
}

/// Solve one leg contact for the current state.
///
/// The tyre model is a friction-limited contact:
///
/// * normal: spring + damper with a stiff mechanical stop beyond
///   `max_compression_m`;
/// * longitudinal: rolling resistance plus brake force limited by
///   `mu_long·N`;
/// * lateral: cornering stiffness on the slip angle, saturating at
///   `mu_lat·N` (full stick at standstill), scaled together with the
///   longitudinal force onto a friction ellipse.
pub fn solve_leg(
    leg: &mut GearLeg,
    state: &RigidBodyState,
    ground: &dyn GroundProvider,
    controls: &ControlInputs,
) -> GearContact {
    let cfg = leg.config;
    let patch_body = cfg.attach_body + DVec3::new(0.0, 0.0, cfg.strut_length_m);
    let patch_world = state.position_m + state.orientation * patch_body;
    let surface = ground.surface_at(patch_world.x, patch_world.y);
    let normal = surface.normal.normalize();

    // Signed distance of the patch below the surface (positive = contact).
    // For the vertical-plane case this is `height - z`; for tilted planes
    // use the plane through (0, 0, height) with the given normal.
    let depth = if (normal.z - 1.0).abs() < 1e-9 {
        surface.height_m - patch_world.z
    } else {
        (DVec3::new(0.0, 0.0, surface.height_m) - patch_world).dot(normal)
    };

    if depth <= 0.0 {
        leg.compression_m = 0.0;
        leg.normal_load_n = 0.0;
        leg.in_contact = false;
        return GearContact::default();
    }
    leg.in_contact = true;

    let bottomed_out = depth > cfg.max_compression_m;
    let spring_depth = depth.min(cfg.max_compression_m);
    let extra = (depth - cfg.max_compression_m).max(0.0);

    let r_world = state.orientation * patch_body;
    let v_patch =
        state.velocity_mps + state.orientation * state.omega_body_radps.cross(patch_body);
    let v_normal = v_patch.dot(normal);
    let compression_rate = -v_normal; // positive while compressing

    let mut normal_load = cfg.spring_n_per_m * spring_depth
        + 20.0 * cfg.spring_n_per_m * extra // mechanical stop
        + cfg.damper_n_s_per_m * compression_rate;
    if normal_load < 0.0 {
        normal_load = 0.0;
    }
    leg.compression_m = depth;
    leg.normal_load_n = normal_load;

    // Steering: positive command steers right (rotation about body +Z).
    let steer = if cfg.steerable {
        controls.steering.clamp(-1.0, 1.0) * cfg.max_steer_rad
    } else {
        0.0
    };
    let fwd_body = DQuat::from_rotation_z(steer) * DVec3::X;
    let fwd_world = (state.orientation * fwd_body)
        .try_normalize()
        .unwrap_or(DVec3::X);
    // Wheel lateral axis: right-hand orthogonal to (normal, rolling dir).
    let mut lat_world = normal.cross(fwd_world);
    let lat_len = lat_world.length();
    let (lat_world, fwd_world) = if lat_len > 1e-9 {
        lat_world /= lat_len;
        let fwd = lat_world.cross(normal).try_normalize().unwrap_or(fwd_world);
        (lat_world, fwd)
    } else {
        // degenerate alignment with the normal: any tangent pair works
        (DVec3::Y, DVec3::X)
    };

    let v_tangential = v_patch - normal * v_normal;
    let v_long = v_tangential.dot(fwd_world);
    let v_lat = v_tangential.dot(lat_world);

    // Longitudinal: rolling resistance + brakes, blended through zero to
    // avoid dithering at standstill.
    let brake = controls.brake.clamp(0.0, 1.0);
    let brake_frac = if cfg.braked { brake } else { 0.0 };
    let eps_v = 0.05f64;
    let sign_smooth = (v_long / eps_v).tanh();
    let mut f_long =
        -(cfg.rolling_resistance + brake_frac * cfg.mu_long) * normal_load * sign_smooth;

    // Lateral: slip angle opposes the patch's sideways velocity; cornering
    // stiffness is linear near zero slip and saturates at mu_lat·N.
    let slip_angle = v_lat.atan2(v_long.abs().max(eps_v * 0.2));
    let mut f_lat = -cfg.cornering_stiffness * slip_angle;
    let mu_lat_max = cfg.mu_lat * normal_load;
    f_lat = f_lat.clamp(-mu_lat_max, mu_lat_max);

    // Friction ellipse coupling.
    let mu_long_max = (cfg.mu_long * normal_load).max(1e-6);
    let ellipse = (f_long / mu_long_max).hypot(f_lat / mu_lat_max.max(1e-6));
    if ellipse > 1.0 {
        f_long /= ellipse;
        f_lat /= ellipse;
    }

    let force_world = normal * normal_load + fwd_world * f_long + lat_world * f_lat;
    let torque_body = state.orientation.conjugate() * r_world.cross(force_world);

    GearContact { force_world, torque_body, bottomed_out }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leg_config() -> GearLegConfig {
        GearLegConfig::tricycle(
            DVec3::new(0.5, 0.0, 0.4),
            0.5,
            60_000.0,
            4_000.0,
            0.25,
            0.3,
            GearAxle::Custom { steerable: true, braked: true },
        )
    }

    #[test]
    fn plane_ground_samples() {
        let g = PlaneGround::new(12.5);
        let s = g.surface_at(1e6, -1e6);
        assert_eq!(s.height_m, 12.5);
        assert_eq!(s.normal, DVec3::Z);
    }

    #[test]
    fn airborne_leg_produces_no_force() {
        let mut leg = GearLeg::new(leg_config());
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, 100.0),
            ..Default::default()
        };
        let c = solve_leg(&mut leg, &state, &PlaneGround::new(0.0), &ControlInputs::default());
        assert_eq!(c.force_world, DVec3::ZERO);
        assert!(!leg.in_contact);
    }

    #[test]
    fn static_support_matches_weight() {
        // patch at z = −depth → compression = depth
        let k = 60_000.0f64;
        let mg = 1000.0 * crate::G0;
        let depth = mg / k; // equilibrium compression
        let mut leg = GearLeg::new(leg_config());
        // patch_body.z = 0.4 + 0.5 = 0.9; place state so patch sits `depth`
        // below a ground plane at 0: patch_world.z = −depth
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, -depth - 0.9),
            ..Default::default()
        };
        let c = solve_leg(&mut leg, &state, &PlaneGround::new(0.0), &ControlInputs::default());
        assert!(c.force_world.z > 0.0);
        assert!((c.force_world.z - mg).abs() < 1e-6, "{}", c.force_world.z);
        assert!(leg.in_contact);
        assert!(!c.bottomed_out);
    }

    #[test]
    fn bottoming_stop_engages_beyond_travel() {
        let mut leg = GearLeg::new(leg_config());
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, -2.0), // patch 1.1 m below ground
            ..Default::default()
        };
        let c = solve_leg(&mut leg, &state, &PlaneGround::new(0.0), &ControlInputs::default());
        assert!(c.bottomed_out);
        // mechanical stop carries far more than the plain spring travel
        let plain = 60_000.0 * leg.config.max_compression_m;
        assert!(c.force_world.z > plain, "{} <= {plain}", c.force_world.z);
    }

    #[test]
    fn lateral_force_saturates_at_friction_limit() {
        let mut leg = GearLeg::new(leg_config());
        let depth = 0.05f64;
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, -depth - 0.9),
            velocity_mps: DVec3::new(0.0, 20.0, 0.0), // huge sideslip
            ..Default::default()
        };
        let c = solve_leg(&mut leg, &state, &PlaneGround::new(0.0), &ControlInputs::default());
        let n = leg.normal_load_n;
        let f_lat = c.force_world.y; // lat axis at zero heading/steer is +Y
        assert!(f_lat.abs() <= leg.config.mu_lat * n + 1e-6, "{f_lat} vs {}", leg.config.mu_lat * n);
        assert!(f_lat < 0.0, "must oppose the +Y slip: {f_lat}");
    }

    #[test]
    fn braking_slows_forward_roll() {
        let mut leg = GearLeg::new(leg_config());
        let depth = 0.05f64;
        let mk = |brake: f64, leg: &mut GearLeg| {
            let state = RigidBodyState {
                position_m: DVec3::new(0.0, 0.0, -depth - 0.9),
                velocity_mps: DVec3::new(15.0, 0.0, 0.0),
                ..Default::default()
            };
            let controls = ControlInputs { brake, ..Default::default() };
            solve_leg(leg, &state, &PlaneGround::new(0.0), &controls)
        };
        let rolling = mk(0.0, &mut leg);
        let braking = mk(1.0, &mut leg);
        assert!(braking.force_world.x < rolling.force_world.x, "brake must pull back");
        assert!(braking.force_world.x < 0.0);
        // rolling resistance alone must also oppose motion
        assert!(rolling.force_world.x < 0.0);
    }

    #[test]
    fn steering_aims_the_wheel() {
        let mut leg = GearLeg::new(leg_config());
        let depth = 0.05f64;
        let state = RigidBodyState {
            position_m: DVec3::new(0.0, 0.0, -depth - 0.9),
            velocity_mps: DVec3::new(10.0, 0.0, 0.0),
            ..Default::default()
        };
        let controls = ControlInputs { steering: 1.0, ..Default::default() };
        let c = solve_leg(&mut leg, &state, &PlaneGround::new(0.0), &controls);
        // rolling forward with the wheel steered right: tyre forces must
        // include a rightward (+Y) component
        assert!(c.force_world.y > 0.0, "steered tyre must push right: {}", c.force_world.y);
    }
}

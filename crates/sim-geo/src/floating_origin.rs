//! Camera-relative floating origin (the 64-bit world → 32-bit GPU bridge).
//!
//! All planetary coordinates live in 64-bit ECEF. GPU buffers must be
//! 32-bit: the *only* sanctioned conversion is to subtract the camera's
//! `f64` ECEF position from the entity's `f64` ECEF position **on the
//! CPU**, cast the difference to `f32`, and only then upload. This module
//! provides [`FloatingOrigin`], the carrier of the camera's double
//! position, and its conversion helpers.

use crate::ellipsoid::Ellipsoid;
use crate::types::{Ecef, LatLonAlt};
use glam::{DQuat, DVec3, Mat4, Quat, Vec3};

fn f32v(v: DVec3) -> Vec3 {
    Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}

fn f64v(v: Vec3) -> DVec3 {
    DVec3::new(v.x as f64, v.y as f64, v.z as f64)
}

fn quat_f32(q: DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32).normalize()
}

/// Holds the camera's earth-fixed position and converts world coordinates
/// into camera-relative 32-bit coordinates for rendering.
#[derive(Debug, Clone, Copy)]
pub struct FloatingOrigin {
    camera_ecef: DVec3,
    camera_quat: DQuat,
}

impl FloatingOrigin {
    /// Create a floating origin at the given ECEF camera position with the
    /// given camera orientation (ECEF quaternion).
    pub fn new(camera_ecef: Ecef, camera_quat: DQuat) -> Self {
        Self { camera_ecef: camera_ecef.as_dvec3(), camera_quat: camera_quat.normalize() }
    }

    /// Create a floating origin from a geodetic camera position.
    pub fn from_geodetic(origin: LatLonAlt, quat: DQuat, ellipsoid: Ellipsoid) -> Self {
        Self::new(ellipsoid.geodetic_to_ecef(&origin), quat)
    }

    /// Current camera ECEF position (metres, f64).
    pub fn camera_ecef(&self) -> Ecef {
        Ecef::from(self.camera_ecef)
    }

    /// Move the origin (typically once per rendered frame).
    pub fn set_camera_ecef(&mut self, p: Ecef) {
        self.camera_ecef = p.as_dvec3();
    }

    /// Set the camera orientation (ECEF quaternion).
    pub fn set_camera_quat(&mut self, q: DQuat) {
        self.camera_quat = q.normalize();
    }

    /// Camera orientation (ECEF quaternion).
    pub fn camera_quat(&self) -> DQuat {
        self.camera_quat
    }

    /// Convert an ECEF world position to a **camera-relative `f32`**
    /// position in metres: `(world - camera) as f32`, performed in `f64`
    /// first and narrowed afterwards.
    pub fn to_camera_relative(&self, world: Ecef) -> Vec3 {
        f32v(world.as_dvec3() - self.camera_ecef)
    }

    /// Camera-relative position in **kilometres** (used by large-scale
    /// terrain patches whose float precision budget needs km units).
    pub fn to_camera_relative_km(&self, world: Ecef) -> Vec3 {
        f32v((world.as_dvec3() - self.camera_ecef) / 1000.0)
    }

    /// Convert ECEF → camera-relative f32 in the camera's own basis:
    /// `(world - camera)` in `f64`, rotated by the inverse camera
    /// orientation, then narrowed to `f32`. Result is expressed with
    /// +X right, +Y up, −Z forward (the convention expected by the GPU
    /// camera uniform).
    pub fn to_view_space(&self, world: Ecef) -> Vec3 {
        let rel = world.as_dvec3() - self.camera_ecef;
        f32v(self.camera_quat.inverse() * rel)
    }

    /// Build the 32-bit view matrix. Entity positions passed to the GPU
    /// are already camera-relative, so only the inverse camera rotation
    /// remains.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_quat(quat_f32(self.camera_quat.inverse()))
    }

    /// Model matrix placing an object at a camera-relative position with
    /// orientation and scale.
    pub fn model_matrix(&self, relative_pos: Vec3, quat: Quat, scale: Vec3) -> Mat4 {
        Mat4::from_scale_rotation_translation(scale, quat, relative_pos)
    }

    /// Cast a double ECEF delta to f32 relative coordinates — the
    /// primitive used everywhere else in the engine.
    pub fn relative_from_delta(delta: DVec3) -> Vec3 {
        f32v(delta)
    }

    /// Nearest world ECEF for a given relative vector (f32 → f64), used
    /// for picking/debug round-trips.
    pub fn to_world(&self, relative: Vec3) -> Ecef {
        Ecef::from(self.camera_ecef + f64v(relative))
    }

    /// Compute an object's position relative to *this* origin from a
    /// position relative to *another* origin (multi-origin streaming).
    /// The rebasing arithmetic happens in `f64`; only the final
    /// difference is narrowed to `f32`.
    pub fn rebase(&self, other: &FloatingOrigin, rel_to_other: Ecef) -> Vec3 {
        let obj = rel_to_other.as_dvec3() + other.camera_ecef;
        f32v(obj - self.camera_ecef)
    }

    /// Distance in metres from the camera to an ECEF point, computed in
    /// 64-bit (precision-preserving).
    pub fn distance_to(&self, world: Ecef) -> f64 {
        (world.as_dvec3() - self.camera_ecef).length()
    }

    /// True when the camera sits inside `radius_m` of the world point.
    pub fn is_within(&self, world: Ecef, radius_m: f64) -> bool {
        self.distance_to(world) <= radius_m
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LatLonAlt, WGS84};

    #[test]
    fn relative_position_is_plain_subtraction_in_f64() {
        let cam = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(40.0, -74.0, 1000.0));
        let fo = FloatingOrigin::new(cam, DQuat::IDENTITY);
        let target = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(40.001, -74.0, 1000.0));
        let rel = fo.to_camera_relative(target);
        let exact = target.as_dvec3() - cam.as_dvec3();
        // f32 has ~7 significant digits: relative distance ~111 m
        let err = (f64v(rel) - exact).length();
        assert!(err < 0.01, "f32 conversion error {err} m");
    }

    #[test]
    fn world_roundtrip_within_f32_tolerance() {
        let cam = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(51.5, -0.12, 50.0));
        let fo = FloatingOrigin::new(cam, DQuat::IDENTITY);
        let obj = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(51.5001, -0.12, 60.0));
        let rel = fo.to_camera_relative(obj);
        let back = fo.to_world(rel);
        let err = (back.as_dvec3() - obj.as_dvec3()).length();
        assert!(err < 0.01, "roundtrip error {err}");
    }

    #[test]
    fn distance_is_computed_in_f64() {
        let cam = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(0.0, 0.0, 0.0));
        let fo = FloatingOrigin::new(cam, DQuat::IDENTITY);
        let far = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(0.0, 1.0, 0.0));
        let d = fo.distance_to(far);
        // 1° of longitude at the equator ≈ 111.319 km
        assert!((d - 111_319.5).abs() < 200.0, "d={d}");
        assert!(fo.is_within(far, d + 1.0));
        assert!(!fo.is_within(far, d - 1.0));
    }

    #[test]
    fn view_matrix_rotates_world_with_camera() {
        // Camera yawed +90° about up: its local +X (right) axis points
        // along world +Y, so a world point at +Y lies on the camera's
        // right and appears at +X in view space.
        let cam = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(0.0, 0.0, 0.0));
        let q = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let fo = FloatingOrigin::new(cam, q);
        let world = Ecef::from(cam.as_dvec3() + DVec3::new(0.0, 100.0, 0.0));
        let rel = fo.to_view_space(world);
        assert!((rel.x - 100.0).abs() < 1e-3, "got {rel}");
        assert!(rel.y.abs() < 1e-3 && rel.z.abs() < 1e-3);
        let v = fo.view_matrix();
        // view matrix must be pure rotation (no translation)
        let t = v.w_axis;
        assert!(t.truncate().length() < 1e-6);
    }

    #[test]
    fn rebasing_between_origins_is_exact_in_f64() {
        let a = FloatingOrigin::new(
            WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(10.0, 10.0, 0.0)),
            DQuat::IDENTITY,
        );
        let b = FloatingOrigin::new(
            WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(10.5, 10.0, 0.0)),
            DQuat::IDENTITY,
        );
        let obj_world = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(10.2, 10.1, 100.0));
        let rel_b = Ecef::from(obj_world.as_dvec3() - b.camera_ecef().as_dvec3());
        let rel_a = a.rebase(&b, rel_b);
        let expect = obj_world.as_dvec3() - a.camera_ecef().as_dvec3();
        assert!((f64v(rel_a) - expect).length() < 0.01);
    }

    #[test]
    fn km_relative_conversion() {
        let cam = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(0.0, 0.0, 0.0));
        let fo = FloatingOrigin::new(cam, DQuat::IDENTITY);
        let p = Ecef::from(cam.as_dvec3() + DVec3::new(2000.0, 0.0, 0.0));
        let km = fo.to_camera_relative_km(p);
        assert!((km.x - 2.0).abs() < 1e-5);
    }
}

//! East-North-Up (ENU) local tangent frames.

use crate::ellipsoid::Ellipsoid;
use crate::types::{Ecef, Enu, LatLonAlt};
use glam::{DMat3, DQuat, DVec3, EulerRot};

/// A local tangent plane anchored at a geodetic origin.
///
/// The frame is right-handed with axes **east** (+x), **north** (+y),
/// **up** (+z). The ECEF→ENU rotation is carried as a [`DMat3`] whose
/// *rows* are the east/north/up unit vectors expressed in ECEF.
#[derive(Debug, Clone, Copy)]
pub struct EnuFrame {
    origin: LatLonAlt,
    origin_ecef: Ecef,
    rotation: DMat3, // rows: east, north, up (ECEF → ENU)
    ellipsoid: Ellipsoid,
}

impl EnuFrame {
    /// Build the frame for a geodetic origin on `ellipsoid` (usually [`crate::WGS84`]).
    pub fn new(origin: LatLonAlt, ellipsoid: Ellipsoid) -> Self {
        let origin_ecef = ellipsoid.geodetic_to_ecef(&origin);
        let phi = origin.latitude_rad();
        let lam = origin.longitude_rad();
        let (sp, cp) = phi.sin_cos();
        let (sl, cl) = lam.sin_cos();
        let east = DVec3::new(-sl, cl, 0.0);
        let north = DVec3::new(-sp * cl, -sp * sl, cp);
        let up = DVec3::new(cp * cl, cp * sl, sp);
        // R has rows east/north/up; DMat3 stores columns, so column j is
        // (east[j], north[j], up[j]).
        let rotation = DMat3::from_cols(
            DVec3::new(east.x, north.x, up.x),
            DVec3::new(east.y, north.y, up.y),
            DVec3::new(east.z, north.z, up.z),
        );
        Self { origin, origin_ecef, rotation, ellipsoid }
    }

    /// Frame origin.
    pub fn origin(&self) -> LatLonAlt {
        self.origin
    }

    /// Frame origin in ECEF.
    pub fn origin_ecef(&self) -> Ecef {
        self.origin_ecef
    }

    /// Reference ellipsoid used by this frame.
    pub fn ellipsoid(&self) -> Ellipsoid {
        self.ellipsoid
    }

    /// ECEF → ENU rotation (rows are east/north/up in ECEF).
    pub fn ecef_to_enu_matrix(&self) -> DMat3 {
        self.rotation
    }

    /// Project an ECEF position into this frame.
    pub fn ecef_to_enu(&self, p: &Ecef) -> Enu {
        let d = p.as_dvec3() - self.origin_ecef.as_dvec3();
        Enu::from(self.rotation * d)
    }

    /// Un-project an ENU offset back to ECEF.
    pub fn enu_to_ecef(&self, e: &Enu) -> Ecef {
        Ecef::from(self.origin_ecef.as_dvec3() + self.rotation.transpose() * e.as_dvec3())
    }

    /// Orientation (ECEF quaternion) of an aircraft whose body axes are
    /// aligned with this ENU frame: identity relative to ENU.
    pub fn enu_identity_ecef_quat(&self) -> DQuat {
        // Columns of Rᵀ are the ENU basis vectors in ECEF; convert the
        // ENU→ECEF rotation matrix to a quaternion.
        mat3_to_quat(self.rotation.transpose())
    }
}

/// Convert a pure rotation matrix to a quaternion (Shepperd's method).
/// `m[row][col]` is indexed in the classical row-major convention.
pub(crate) fn mat3_to_quat(m: DMat3) -> DQuat {
    let k = [
        [m.x_axis.x, m.y_axis.x, m.z_axis.x], // row 0
        [m.x_axis.y, m.y_axis.y, m.z_axis.y], // row 1
        [m.x_axis.z, m.y_axis.z, m.z_axis.z], // row 2
    ];
    let q = {
        let tr = k[0][0] + k[1][1] + k[2][2];
        if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0;
            [
                (k[2][1] - k[1][2]) / s,
                (k[0][2] - k[2][0]) / s,
                (k[1][0] - k[0][1]) / s,
                0.25 * s,
            ]
        } else if k[0][0] > k[1][1] && k[0][0] > k[2][2] {
            let s = (1.0 + k[0][0] - k[1][1] - k[2][2]).sqrt() * 2.0;
            [
                0.25 * s,
                (k[0][1] + k[1][0]) / s,
                (k[0][2] + k[2][0]) / s,
                (k[2][1] - k[1][2]) / s,
            ]
        } else if k[1][1] > k[2][2] {
            let s = (1.0 + k[1][1] - k[0][0] - k[2][2]).sqrt() * 2.0;
            [
                (k[0][1] + k[1][0]) / s,
                0.25 * s,
                (k[1][2] + k[2][1]) / s,
                (k[0][2] - k[2][0]) / s,
            ]
        } else {
            let s = (1.0 + k[2][2] - k[0][0] - k[1][1]).sqrt() * 2.0;
            [
                (k[0][2] - k[2][0]) / s,
                (k[1][2] + k[2][1]) / s,
                0.25 * s,
                (k[1][0] - k[0][1]) / s,
            ]
        }
    };
    DQuat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()
}

/// Aircraft-style yaw/pitch/roll (ZYX intrinsic = heading, elevation, bank)
/// expressed **relative to an ENU frame**, converted to an ECEF quaternion.
pub fn enu_euler_to_ecef_quat(frame: &EnuFrame, yaw: f64, pitch: f64, roll: f64) -> DQuat {
    let body_in_enu = DQuat::from_euler(EulerRot::ZYX, yaw, pitch, roll);
    frame.enu_identity_ecef_quat() * body_in_enu
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WGS84;

    fn is_orthonormal(m: &DMat3, tol: f64) -> bool {
        let i = *m * m.transpose();
        let d = i - DMat3::IDENTITY;
        d.x_axis.abs().max_element() < tol
            && d.y_axis.abs().max_element() < tol
            && d.z_axis.abs().max_element() < tol
    }

    #[test]
    fn rotation_is_orthonormal_and_proper() {
        for (lat, lon) in [(0.0, 0.0), (45.0, -122.0), (-89.0, 30.0), (66.5, 179.0)] {
            let f = EnuFrame::new(LatLonAlt::from_degrees(lat, lon, 0.0), WGS84);
            let r = f.ecef_to_enu_matrix();
            assert!(is_orthonormal(&r, 1e-12), "not orthonormal at {lat},{lon}");
            // proper rotation: det = +1
            let det = r.x_axis.dot(r.y_axis.cross(r.z_axis));
            assert!((det - 1.0).abs() < 1e-12, "det={det}");
        }
    }

    #[test]
    fn origin_projects_to_zero() {
        let f = EnuFrame::new(LatLonAlt::from_degrees(52.0, 13.0, 40.0), WGS84);
        let e = f.ecef_to_enu(&f.origin_ecef());
        assert!(e.as_dvec3().length() < 1e-6);
    }

    #[test]
    fn east_north_up_directions() {
        let f = EnuFrame::new(LatLonAlt::from_degrees(0.0, 0.0, 0.0), WGS84);
        let origin = f.origin_ecef();
        // At (0°N, 0°E): east = +Y ECEF, north = +Z ECEF, up = +X ECEF.
        // Offsets are measured from the frame origin.
        let east = f.ecef_to_enu(&Ecef::from(origin.as_dvec3() + glam::DVec3::new(0.0, 10.0, 0.0)));
        assert!((east.east() - 10.0).abs() < 1e-9);
        assert!(east.north().abs() < 1e-9);
        assert!(east.up().abs() < 1e-9);

        let up = f.ecef_to_enu(&Ecef::from(origin.as_dvec3() + glam::DVec3::new(1_000.0, 0.0, 0.0)));
        assert!((up.up() - 1_000.0).abs() < 1e-6);
        assert!(up.east().abs() < 1e-6 && up.north().abs() < 1e-6);

        let north = f.ecef_to_enu(&Ecef::from(origin.as_dvec3() + glam::DVec3::new(0.0, 0.0, 7.0)));
        assert!((north.north() - 7.0).abs() < 1e-9);
        assert!(north.east().abs() < 1e-9 && north.up().abs() < 1e-9);
    }

    #[test]
    fn roundtrip_ecef_enu() {
        let f = EnuFrame::new(LatLonAlt::from_degrees(-33.9, 151.2, 12.0), WGS84);
        let llh = LatLonAlt::from_degrees(-33.87, 151.1, 85.0);
        let ecef = WGS84.geodetic_to_ecef(&llh);
        let enu = f.ecef_to_enu(&ecef);
        let back = f.enu_to_ecef(&enu);
        assert!((back.as_dvec3() - ecef.as_dvec3()).length() < 1e-6);
    }

    #[test]
    fn up_direction_points_away_from_earth() {
        for (lat, lon) in [(40.0, -74.0), (-45.0, 20.0), (80.0, 100.0)] {
            let f = EnuFrame::new(LatLonAlt::from_degrees(lat, lon, 0.0), WGS84);
            let surface = f.origin_ecef();
            let up = f.ecef_to_enu(&surface).up();
            assert!(up.abs() < 1e-6);
            // 1 km straight up must run exactly along +up of the frame
            let p = f.enu_to_ecef(&Enu::new(0.0, 0.0, 1000.0));
            let vertical = (p.as_dvec3() - surface.as_dvec3()).normalize();
            let up_dir = f.ecef_to_enu_matrix().transpose().col(2);
            assert!((vertical.dot(up_dir) - 1.0).abs() < 1e-12);
            let _ = surface;
        }
    }

    #[test]
    fn enu_identity_quat_rotates_enu_basis_to_ecef_basis() {
        let f = EnuFrame::new(LatLonAlt::from_degrees(35.0, 139.0, 0.0), WGS84);
        let q = f.enu_identity_ecef_quat();
        // ENU x axis (east) in ENU, rotated to ECEF, must equal east row of Rᵀ.
        let east_ecef = q * DVec3::X;
        let r_t = f.ecef_to_enu_matrix().transpose();
        assert!((east_ecef - r_t.col(0)).length() < 1e-9);
        let up_ecef = q * DVec3::Z;
        assert!((up_ecef - r_t.col(2)).length() < 1e-9);
        assert!((q.length() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn euler_yaw_of_90_deg_turns_east_into_north_in_enu() {
        let f = EnuFrame::new(LatLonAlt::from_degrees(10.0, 10.0, 0.0), WGS84);
        let q = enu_euler_to_ecef_quat(&f, std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        // body +X (forward) should point north in ENU coordinates
        let q_enu = f.enu_identity_ecef_quat().inverse() * (q * DVec3::X);
        assert!((q_enu - DVec3::Y).length() < 1e-9, "got {q_enu}");
    }
}

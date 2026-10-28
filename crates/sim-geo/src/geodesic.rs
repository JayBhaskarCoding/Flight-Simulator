//! Spherical great-circle geodesy.
//!
//! These helpers use the WGS84 *authalic* (equal-area) sphere radius from
//! [`crate::Ellipsoid::authalic_radius`], giving ~0.3 % distance accuracy
//! versus the ellipsoid — far better than any downstream consumer needs
//! (tile scheduling, AI traffic routing, weather cell placement).

use crate::ellipsoid::Ellipsoid;
use crate::types::LatLonAlt;
use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// Wrap an angle into (-π, π].
pub fn wrap_pi(rad: f64) -> f64 {
    let a = (rad + PI).rem_euclid(TWO_PI) - PI;
    if a <= -PI {
        PI
    } else {
        a
    }
}

/// Wrap an angle into [0, 2π).
pub fn wrap_two_pi(rad: f64) -> f64 {
    rad.rem_euclid(TWO_PI)
}

/// Great-circle distance between two geodetic points on `ellipsoid`'s
/// authalic sphere, in metres (haversine formula).
pub fn distance_m(a: &LatLonAlt, b: &LatLonAlt, ellipsoid: Ellipsoid) -> f64 {
    let r = ellipsoid.authalic_radius();
    let (phi1, phi2) = (a.latitude_rad(), b.latitude_rad());
    let dphi = phi2 - phi1;
    let dlam = b.longitude_rad() - a.longitude_rad();
    let h = (dphi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (dlam / 2.0).sin().powi(2);
    2.0 * r * h.sqrt().asin()
}

/// Initial great-circle bearing from `a` to `b`, radians clockwise from
/// true north ∈ [0, 2π).
pub fn initial_bearing(a: &LatLonAlt, b: &LatLonAlt) -> f64 {
    let (phi1, phi2) = (a.latitude_rad(), b.latitude_rad());
    let dlam = b.longitude_rad() - a.longitude_rad();
    let y = dlam.sin() * phi2.cos();
    let x = phi1.cos() * phi2.sin() - phi1.sin() * phi2.cos() * dlam.cos();
    wrap_two_pi(y.atan2(x))
}

/// Interpolate a point `t ∈ [0, 1]` along the great circle from `a` to `b`
/// (slerp on the unit sphere; scale-invariant, so no ellipsoid is needed).
pub fn interpolate(a: &LatLonAlt, b: &LatLonAlt, t: f64) -> LatLonAlt {
    let to_unit = |p: &LatLonAlt| -> [f64; 3] {
        let (s, c) = p.latitude_rad().sin_cos();
        let (sl, cl) = p.longitude_rad().sin_cos();
        [c * cl, c * sl, s]
    };
    let va = to_unit(a);
    let vb = to_unit(b);
    let dot = (va[0] * vb[0] + va[1] * vb[1] + va[2] * vb[2]).clamp(-1.0, 1.0);
    let omega = dot.acos();
    let alt = a.altitude_m() * (1.0 - t) + b.altitude_m() * t;
    if omega < 1e-12 {
        return LatLonAlt::new(a.latitude_rad(), a.longitude_rad(), alt);
    }
    // standard spherical linear interpolation (slerp)
    let denom = omega.sin();
    let (k0, k1) = if denom.abs() < 1e-12 {
        (1.0 - t, t)
    } else {
        (((1.0 - t) * omega).sin() / denom, (t * omega).sin() / denom)
    };
    let v = [
        va[0] * k0 + vb[0] * k1,
        va[1] * k0 + vb[1] * k1,
        va[2] * k0 + vb[2] * k1,
    ];
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let lat = (v[2] / len).asin();
    let lon = v[1].atan2(v[0]);
    LatLonAlt::new(lat, lon, alt)
}

/// Destination point given start, initial bearing (rad) and distance (m)
/// on the authalic sphere.
pub fn destination(a: &LatLonAlt, bearing_rad: f64, distance: f64, ellipsoid: Ellipsoid) -> LatLonAlt {
    let r = ellipsoid.authalic_radius();
    let delta = distance / r;
    let (phi1, lam1) = (a.latitude_rad(), a.longitude_rad());
    let (sd, cd) = delta.sin_cos();
    let (sp, cp) = phi1.sin_cos();
    let (sb, cb) = bearing_rad.sin_cos();
    let phi2 = (sp * cd + cp * sd * cb).asin();
    let lam2 = lam1 + (sd * sb).atan2(cp * cd - sp * sd * cb);
    LatLonAlt::new(phi2, wrap_pi(lam2), a.altitude_m())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WGS84;

    #[test]
    fn wrap_ranges() {
        assert!((wrap_pi(3.0 * PI) - PI).abs() < 1e-12);
        assert!((wrap_pi(-3.0 * PI) - PI).abs() < 1e-12 || (wrap_pi(-3.0 * PI) + PI).abs() < 1e-12);
        assert!((wrap_pi(0.5) - 0.5).abs() < 1e-15);
        assert!((wrap_two_pi(-0.25) - (TWO_PI - 0.25)).abs() < 1e-12);
    }

    #[test]
    fn london_to_paris() {
        let lhr = LatLonAlt::from_degrees(51.4700, -0.4543, 0.0);
        let cdg = LatLonAlt::from_degrees(49.0097, 2.5479, 0.0);
        let d = distance_m(&lhr, &cdg, WGS84);
        // great-circle LHR→CDG ≈ 348 km
        assert!((d - 348_000.0).abs() < 5_000.0, "d={d}");
        let b = initial_bearing(&lhr, &cdg);
        // roughly south-east
        assert!(b.to_degrees() > 100.0 && b.to_degrees() < 160.0, "bearing {}", b.to_degrees());
    }

    #[test]
    fn one_degree_at_equator() {
        let a = LatLonAlt::from_degrees(0.0, 0.0, 0.0);
        let b = LatLonAlt::from_degrees(0.0, 1.0, 0.0);
        let d = distance_m(&a, &b, WGS84);
        // authalic sphere: 2πR/360 ≈ 111.19 km
        assert!((d - 111_194.9).abs() < 50.0, "d={d}");
    }

    #[test]
    fn antipodal_distance_is_half_circumference() {
        let a = LatLonAlt::from_degrees(10.0, 20.0, 0.0);
        let b = LatLonAlt::from_degrees(-10.0, 200.0, 0.0);
        let d = distance_m(&a, &b, WGS84);
        let half = PI * WGS84.authalic_radius();
        assert!((d - half).abs() < 1000.0, "d={d} vs {half}");
    }

    #[test]
    fn destination_matches_distance() {
        let start = LatLonAlt::from_degrees(48.8566, 2.3522, 0.0);
        let bearing = 60.0f64.to_radians();
        let d = 50_000.0;
        let p = destination(&start, bearing, d, WGS84);
        let back = distance_m(&start, &p, WGS84);
        assert!((back - d).abs() < 1.0, "back={back}");
        // bearing from start to p should equal requested bearing
        let b = initial_bearing(&start, &p);
        assert!((b - bearing).abs() < 1e-6, "b={b}");
    }

    #[test]
    fn interpolate_midpoint_is_halfway() {
        let a = LatLonAlt::from_degrees(0.0, 0.0, 100.0);
        let b = LatLonAlt::from_degrees(0.0, 90.0, 200.0);
        let mid = interpolate(&a, &b, 0.5);
        assert!((mid.latitude_deg()).abs() < 1e-9);
        assert!((mid.longitude_deg() - 45.0).abs() < 1e-9);
        assert!((mid.altitude_m() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn interpolate_endpoints() {
        let a = LatLonAlt::from_degrees(35.0, 139.0, 0.0);
        let b = LatLonAlt::from_degrees(-33.9, 151.2, 0.0);
        let p0 = interpolate(&a, &b, 0.0);
        assert!((p0.latitude_deg() - 35.0).abs() < 1e-9);
        let p1 = interpolate(&a, &b, 1.0);
        assert!((p1.latitude_deg() + 33.9).abs() < 1e-6);
        assert!((p1.longitude_deg() - 151.2).abs() < 1e-6);
    }
}

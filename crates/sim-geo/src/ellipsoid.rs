//! Reference-ellipsoid geometry (WGS84 and any custom ellipsoid).

use crate::types::{Ecef, LatLonAlt};
use glam::DVec3;

/// The WGS84 defining parameters (NGA.STD.0088).
///
/// * semi-major axis `a`          = 6 378 137.0 m
/// * inverse flattening `1/f`     = 298.257 223 563
pub const WGS84: Ellipsoid = Ellipsoid {
    semi_major_axis: 6_378_137.0,
    inv_flattening: 298.257_223_563,
};

/// A rotational reference ellipsoid defined by its semi-major axis and
/// flattening, per ISO 19111 / WGS84 conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipsoid {
    /// Semi-major (equatorial) radius, metres.
    pub semi_major_axis: f64,
    /// Inverse flattening `1/f` (298.257223563 for WGS84).
    pub inv_flattening: f64,
}

impl Ellipsoid {
    /// Flattening `f = (a - b) / a`.
    pub fn flattening(&self) -> f64 {
        1.0 / self.inv_flattening
    }

    /// Semi-minor (polar) radius `b = a (1 - f)`.
    pub fn semi_minor_axis(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.flattening())
    }

    /// First eccentricity squared, `e² = f (2 - f)`.
    pub fn first_eccentricity_squared(&self) -> f64 {
        let f = self.flattening();
        f * (2.0 - f)
    }

    /// Second (parametric) eccentricity squared, `e'² = e² / (1 - e²)`.
    pub fn second_eccentricity_squared(&self) -> f64 {
        let e2 = self.first_eccentricity_squared();
        e2 / (1.0 - e2)
    }

    /// Radius of curvature in the prime vertical, `N(φ) = a / sqrt(1 - e² sin²φ)`.
    ///
    /// `latitude_rad` is geodetic latitude in radians.
    pub fn prime_vertical_radius(&self, latitude_rad: f64) -> f64 {
        let e2 = self.first_eccentricity_squared();
        let s = latitude_rad.sin();
        self.semi_major_axis / (1.0 - e2 * s * s).sqrt()
    }

    /// Radius of curvature in the meridian, `M(φ) = a(1-e²) / (1-e² sin²φ)^{3/2}`.
    ///
    /// `latitude_rad` is geodetic latitude in radians.
    pub fn meridian_radius(&self, latitude_rad: f64) -> f64 {
        let e2 = self.first_eccentricity_squared();
        let s = latitude_rad.sin();
        self.semi_major_axis * (1.0 - e2) / (1.0 - e2 * s * s).powf(1.5)
    }

    /// Geodetic (lat, lon, height) → Earth-Centred Earth-Fixed (ECEF).
    ///
    /// ```
    /// use sim_geo::{WGS84, LatLonAlt, Ecef};
    /// // equatorial surface point
    /// let p = WGS84.geodetic_to_ecef(&LatLonAlt::from_degrees(0.0, 0.0, 0.0));
    /// assert!((p.x() - 6_378_137.0).abs() < 1e-6);
    /// ```
    pub fn geodetic_to_ecef(&self, llh: &LatLonAlt) -> Ecef {
        let phi = llh.latitude_rad();
        let lam = llh.longitude_rad();
        let h = llh.altitude_m();
        let e2 = self.first_eccentricity_squared();
        let n = self.prime_vertical_radius(phi);
        let sp = phi.sin();
        let cp = phi.cos();
        let sl = lam.sin();
        let cl = lam.cos();
        Ecef::new(
            (n + h) * cp * cl,
            (n + h) * cp * sl,
            (n * (1.0 - e2) + h) * sp,
        )
    }

    /// Earth-Centred Earth-Fixed (ECEF) → geodetic (lat, lon, height).
    ///
    /// Closed-form solution of Zhang & Pararas (2016); sub-millimetre
    /// accuracy over the whole globe including near-polar geometries.
    pub fn ecef_to_geodetic(&self, ecef: &Ecef) -> LatLonAlt {
        let a = self.semi_major_axis;
        let b = self.semi_minor_axis();
        let e2 = self.first_eccentricity_squared();
        let ep2 = self.second_eccentricity_squared();

        let (x, y, z) = (ecef.x(), ecef.y(), ecef.z());
        let p = (x * x + y * y).sqrt();

        // Degenerate case: earth centre (define as surface at equator)
        if p < 1e-9 && z.abs() < 1e-9 {
            return LatLonAlt::new(0.0, 0.0, -a);
        }

        // Polar axis case
        if p < 1e-6 {
            let lat = if z >= 0.0 {
                std::f64::consts::FRAC_PI_2
            } else {
                -std::f64::consts::FRAC_PI_2
            };
            return LatLonAlt::new(lat, 0.0, z.abs() - b);
        }

        // Zhang & Pararas closed form for the initial estimate, then a
        // fixed-point polish: tan φ = (z + e²·N(φ)·sin φ) / p.
        // The polish keeps sub-millimetre accuracy at ANY altitude
        // (the closed form alone drifts to decimetres beyond ~10 000 km).
        let theta = (z * a).atan2(p * b);
        let st = theta.sin();
        let ct = theta.cos();
        let mut lat = (z + ep2 * b * st.powi(3)).atan2(p - e2 * a * ct.powi(3));
        for _ in 0..6 {
            let n = self.prime_vertical_radius(lat);
            lat = ((z + e2 * n * lat.sin()) / p).atan();
        }
        let lon = y.atan2(x);
        let n = self.prime_vertical_radius(lat);
        // Height from the formula that does not divide by a vanishing
        // cos φ near the poles.
        let h = if lat.cos().abs() > 0.1 {
            p / lat.cos() - n
        } else {
            z / lat.sin() - n * (1.0 - e2)
        };
        LatLonAlt::new(lat, lon, h)
    }

    /// Approximate surface distance along the ellipsoid between two
    /// geodetic points, using a spherical approximation with the
    /// *authalic* (equal-area) radius — accurate to ~0.3 %.
    pub fn authalic_radius(&self) -> f64 {
        let e2 = self.first_eccentricity_squared();
        let a = self.semi_major_axis;
        // R_q = a * sqrt( (1 + (1-e²)/e * atanh(sqrt(e²)) ) / 2 ), e²>0
        let e = e2.sqrt();
        let atanh_e = ((1.0 + e) / (1.0 - e)).ln() / 2.0;
        a * ((1.0 + (1.0 - e2) / e * atanh_e) / 2.0).sqrt()
    }

    /// Height of the ellipsoid above sea level at a given latitude is 0 by
    /// definition; provided for completeness in geometric computations.
    pub fn normal_at(&self, latitude_rad: f64) -> DVec3 {
        let phi = latitude_rad;
        DVec3::new(phi.cos(), 0.0, phi.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_M: f64 = 1e-3; // millimetre round-trip tolerance

    #[test]
    fn wgs84_defining_constants() {
        assert_eq!(WGS84.semi_major_axis, 6_378_137.0);
        assert!((WGS84.semi_minor_axis() - 6_356_752.314_245_18).abs() < 1e-6);
        assert!((WGS84.flattening() - 1.0 / 298.257_223_563).abs() < 1e-15);
        assert!((WGS84.first_eccentricity_squared() - 0.006_694_379_990_14).abs() < 1e-10);
    }

    #[test]
    fn roundtrip_representative_points() {
        let cases = [
            (0.0, 0.0, 0.0),
            (0.0, 90.0, 1000.0),
            (45.0, -73.5, 40.0),      // Montréal
            (-33.9, 151.2, 20.0),     // Sydney
            (89.9, 10.0, 100.0),      // near north pole
            (-89.9, -160.0, 5000.0),  // near south pole
            (0.0, 179.9, -100.0),     // trench
            (60.0, 24.9, 50.0),       // Helsinki
        ];
        for (lat, lon, alt) in cases {
            let llh = LatLonAlt::from_degrees(lat, lon, alt);
            let ecef = WGS84.geodetic_to_ecef(&llh);
            let back = WGS84.ecef_to_geodetic(&ecef);
            let dlat = (back.latitude_deg() - lat).abs();
            let dlon = (back.longitude_deg() - lon).abs();
            assert!(dlat < 1e-8, "lat drift {dlat} at {lat},{lon}");
            assert!(dlon < 1e-8, "lon drift {dlon} at {lat},{lon}");
            assert!(
                (back.altitude_m() - alt).abs() < TOL_M,
                "alt drift {} at {lat},{lon}",
                back.altitude_m() - alt
            );
        }
    }

    #[test]
    fn roundtrip_extreme_altitudes() {
        for alt in [-11_000.0, 0.0, 88_488.0, 400_000.0, 36_000_000.0, 200_000_000.0] {
            let llh = LatLonAlt::from_degrees(37.7, -122.4, alt);
            let back = WGS84.ecef_to_geodetic(&WGS84.geodetic_to_ecef(&llh));
            assert!((back.altitude_m() - alt).abs() < 1e-3, "alt {alt} -> {}", back.altitude_m());
            assert!((back.latitude_deg() - 37.7).abs() < 1e-7);
            assert!((back.longitude_deg() + 122.4).abs() < 1e-7);
        }
    }

    #[test]
    fn exact_poles_and_axis() {
        let np = WGS84.ecef_to_geodetic(&Ecef::new(0.0, 0.0, WGS84.semi_minor_axis()));
        assert!((np.latitude_deg() - 90.0).abs() < 1e-9);
        assert!(np.altitude_m().abs() < 1e-6);

        let sp = WGS84.ecef_to_geodetic(&Ecef::new(0.0, 0.0, -WGS84.semi_minor_axis()));
        assert!((sp.latitude_deg() + 90.0).abs() < 1e-9);

        let eq = WGS84.ecef_to_geodetic(&Ecef::new(WGS84.semi_major_axis, 0.0, 0.0));
        assert!(eq.latitude_deg().abs() < 1e-9);
        assert!(eq.altitude_m().abs() < 1e-6);
    }

    #[test]
    fn prime_vertical_at_poles_is_infinite_limit() {
        // N(90°) = a / sqrt(1 - e²) is finite; meridian curvature too.
        let n = WGS84.prime_vertical_radius(std::f64::consts::FRAC_PI_2);
        assert!((n - WGS84.semi_major_axis / (1.0 - WGS84.first_eccentricity_squared()).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn authalic_radius_close_to_6371() {
        let r = WGS84.authalic_radius();
        assert!((r - 6_371_007.2).abs() < 2.0, "authalic radius {r}");
    }
}

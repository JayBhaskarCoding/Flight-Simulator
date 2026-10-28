//! Core coordinate newtypes. Everything here is 64-bit (`f64` / `DVec3`).

use glam::DVec3;
use std::ops::{Add, AddAssign, Deref, Div, Mul, Neg, Sub, SubAssign};

/// Geodetic position on (or above) the reference ellipsoid.
///
/// Internally stored in **radians** and metres; degree-based constructors
/// and accessors are provided because every aviation data source
/// (`flight_model.cfg`, METAR, SimVars) uses degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatLonAlt {
    lat_rad: f64,
    lon_rad: f64,
    alt_m: f64,
}

impl LatLonAlt {
    /// Construct from radians (latitude ∈ [-π/2, π/2], longitude ∈ [-π, π])
    /// and altitude in metres above the ellipsoid.
    pub fn new(lat_rad: f64, lon_rad: f64, alt_m: f64) -> Self {
        Self { lat_rad, lon_rad, alt_m }
    }

    /// Construct from degrees and metres above the ellipsoid.
    pub fn from_degrees(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Self {
        Self {
            lat_rad: lat_deg.to_radians(),
            lon_rad: lon_deg.to_radians(),
            alt_m,
        }
    }

    /// Construct from degrees and feet above the ellipsoid
    /// (aviation data sources report altitude in feet).
    pub fn from_degrees_feet(lat_deg: f64, lon_deg: f64, alt_ft: f64) -> Self {
        Self::from_degrees(lat_deg, lon_deg, alt_ft * 0.3048)
    }

    pub fn latitude_rad(&self) -> f64 {
        self.lat_rad
    }
    pub fn longitude_rad(&self) -> f64 {
        self.lon_rad
    }
    pub fn altitude_m(&self) -> f64 {
        self.alt_m
    }
    pub fn latitude_deg(&self) -> f64 {
        self.lat_rad.to_degrees()
    }
    pub fn longitude_deg(&self) -> f64 {
        self.lon_rad.to_degrees()
    }
    pub fn altitude_ft(&self) -> f64 {
        self.alt_m / 0.3048
    }

    /// Returns a copy with a new latitude (radians).
    pub fn with_latitude_rad(mut self, lat_rad: f64) -> Self {
        self.lat_rad = lat_rad;
        self
    }
    /// Returns a copy with a new longitude (radians).
    pub fn with_longitude_rad(mut self, lon_rad: f64) -> Self {
        self.lon_rad = lon_rad;
        self
    }
    /// Returns a copy with a new altitude (metres).
    pub fn with_altitude_m(mut self, alt_m: f64) -> Self {
        self.alt_m = alt_m;
        self
    }
}

impl std::fmt::Display for LatLonAlt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:.6}°, {:.6}°, {:.1} m",
            self.latitude_deg(),
            self.longitude_deg(),
            self.alt_m
        )
    }
}

macro_rules! vec_newtype_ops {
    ($name:ident) => {
        impl From<DVec3> for $name {
            fn from(v: DVec3) -> Self {
                $name(v)
            }
        }
        impl $name {
            /// Raw ECEF/ENU components.
            pub fn x(&self) -> f64 {
                self.0.x
            }
            pub fn y(&self) -> f64 {
                self.0.y
            }
            pub fn z(&self) -> f64 {
                self.0.z
            }
            pub fn as_dvec3(&self) -> DVec3 {
                self.0
            }
        }
        impl Deref for $name {
            type Target = DVec3;
            fn deref(&self) -> &DVec3 {
                &self.0
            }
        }
        impl Add for $name {
            type Output = $name;
            fn add(self, rhs: $name) -> $name {
                $name(self.0 + rhs.0)
            }
        }
        impl Sub for $name {
            type Output = $name;
            fn sub(self, rhs: $name) -> $name {
                $name(self.0 - rhs.0)
            }
        }
        impl AddAssign for $name {
            fn add_assign(&mut self, rhs: $name) {
                self.0 += rhs.0;
            }
        }
        impl SubAssign for $name {
            fn sub_assign(&mut self, rhs: $name) {
                self.0 -= rhs.0;
            }
        }
        impl Neg for $name {
            type Output = $name;
            fn neg(self) -> $name {
                $name(-self.0)
            }
        }
        impl Mul<f64> for $name {
            type Output = $name;
            fn mul(self, rhs: f64) -> $name {
                $name(self.0 * rhs)
            }
        }
        impl Div<f64> for $name {
            type Output = $name;
            fn div(self, rhs: f64) -> $name {
                $name(self.0 / rhs)
            }
        }
    };
}

/// Earth-Centred Earth-Fixed (ECEF) position, metres, WGS84.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Ecef(DVec3);

/// Local east-north-up (ENU) offset from a frame origin, metres.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Enu(DVec3);

impl Ecef {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self(DVec3::new(x, y, z))
    }
    pub fn origin() -> Self {
        Self(DVec3::ZERO)
    }
}

impl Enu {
    /// Construct from (east, north, up) components in metres.
    pub fn new(east: f64, north: f64, up: f64) -> Self {
        Self(DVec3::new(east, north, up))
    }
    pub fn east(&self) -> f64 {
        self.0.x
    }
    pub fn north(&self) -> f64 {
        self.0.y
    }
    pub fn up(&self) -> f64 {
        self.0.z
    }
}

vec_newtype_ops!(Ecef);
vec_newtype_ops!(Enu);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_conversions_roundtrip() {
        let p = LatLonAlt::from_degrees(51.4775, -0.4614, 25.0);
        assert!((p.latitude_deg() - 51.4775).abs() < 1e-12);
        assert!((p.longitude_deg() + 0.4614).abs() < 1e-12);
        assert!((p.altitude_ft() - 82.021).abs() < 1e-2);
    }

    #[test]
    fn feet_metres_conversion() {
        let p = LatLonAlt::from_degrees_feet(0.0, 0.0, 1000.0);
        assert!((p.altitude_m() - 304.8).abs() < 1e-9);
    }

    #[test]
    fn newtype_arithmetic() {
        let a = Ecef::new(1.0, 2.0, 3.0);
        let b = Ecef::new(4.0, 5.0, 6.0);
        let d = b - a;
        assert_eq!(d, Ecef::new(3.0, 3.0, 3.0));
        assert_eq!((a + b).x(), 5.0);
        assert_eq!((a * 2.0).z(), 6.0);
        assert_eq!((-a).y(), -2.0);
        let e = Enu::new(1.0, 2.0, 3.0);
        assert_eq!((e.north(), e.east()), (2.0, 1.0));
    }
}

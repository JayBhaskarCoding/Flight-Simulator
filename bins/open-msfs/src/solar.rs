//! Solar position (clean-room NOAA "Solar Calculator" approximation):
//! declination + equation of time from the Julian day, then
//! horizontal coordinates at the observer, expressed in the ENU sky
//! frame (x = east, y = north, z = up) that `sim-atmosphere`'s
//! Y-up model uses (after the ENU→sky mapping in [`super::sim`], the
//! sky frame is y-up — this module returns *astronomical* elevation /
//! azimuth and `sim.rs` maps them).

use std::time::{SystemTime, UNIX_EPOCH};

/// Result of a solar query, in the observer's horizontal frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Elevation above the horizon, rad.
    pub elevation_rad: f64,
    /// Azimuth clockwise from true north, rad (0..2π).
    pub azimuth_rad: f64,
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's
/// `days_from_civil`, shifted to the Unix epoch).
fn days_from_civil_y(y: i64, m: i64, d: i64) -> i64 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Inverse of [`days_from_civil_y`] — (year, month, day).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Day of year (1..=366, fractional within the UTC day).
fn day_of_year(unix_s: f64) -> f64 {
    let epoch_day = (unix_s / 86_400.0).floor() as i64;
    let (y, _, _) = civil_from_days(epoch_day);
    let jan1 = days_from_civil_y(y, 1, 1);
    (epoch_day - jan1 + 1) as f64
}

/// Solar declination, rad (Cooper's equation of the day,
/// `δ = 0.409 · sin(2π(N − 81)/365)`, max error ≈ 0.4°).
fn declination_rad(day_of_year: f64) -> f64 {
    0.409 * (std::f64::consts::TAU * (day_of_year - 81.0) / 365.0).sin()
}

/// Equation of time, minutes (NOAA Almanac polynomial in the day
/// angle `Γ = 2π(N − 1)/365`, max error ≈ 0.5 minutes).
fn equation_of_time_min(day_of_year: f64) -> f64 {
    let gamma = std::f64::consts::TAU * (day_of_year - 1.0) / 365.0;
    229.18 * (0.000_075
        + 0.001_868 * gamma.cos()
        - 0.032_077 * gamma.sin()
        + 0.014_615 * (2.0 * gamma).cos()
        - 0.040_849 * (2.0 * gamma).sin())
}

/// Solar position for `unix_s` at latitude/longitude (radians).
///
/// Local **solar** time is used: `UTC + longitude/15h + EoT`, the
/// astronomically correct hour angle (civil time zones do not shift
/// the Sun — only the observer's longitude does).
pub fn solar_position(unix_s: f64, lat_rad: f64, lon_rad: f64) -> SolarPosition {
    let n = day_of_year(unix_s);
    let decl = declination_rad(n);
    let eot = equation_of_time_min(n);

    let utc_days = (unix_s / 86_400.0).rem_euclid(1.0);
    let utc_hours = utc_days * 24.0;
    let local_hours = (utc_hours + lon_rad.to_degrees() / 15.0 + eot / 60.0).rem_euclid(24.0);

    // Hour angle (0° at local solar noon)
    let ha = (local_hours * 15.0 - 180.0).to_radians();

    // Horizontal coordinates
    let sin_el = lat_rad.sin() * decl.sin() + lat_rad.cos() * decl.cos() * ha.cos();
    let elevation = sin_el.clamp(-1.0, 1.0).asin();
    let sin_az = -ha.sin() * decl.cos();
    let cos_az = lat_rad.cos() * decl.sin() - lat_rad.sin() * decl.cos() * ha.cos();
    let azimuth = sin_az.atan2(cos_az).rem_euclid(std::f64::consts::TAU);

    SolarPosition { elevation_rad: elevation, azimuth_rad: azimuth }
}

/// Convenience: current wall-clock UTC seconds since the epoch.
pub fn unix_now() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEG: f64 = std::f64::consts::PI / 180.0;

    #[test]
    fn sun_is_above_the_horizon_at_local_noon_equinox() {
        // 2024-03-20 ~12:00 UTC at the equator/prime meridian:
        // declination ≈ 0, hour angle ≈ 0 → elevation ≈ 90°.
        let unix = 1_710_936_000.0; // 2024-03-20 12:00 UTC
        let p = solar_position(unix, 0.0, 0.0);
        assert!(
            p.elevation_rad > 70.0 * DEG && p.elevation_rad < 90.0 * DEG,
            "elevation {}°",
            p.elevation_rad.to_degrees()
        );
    }

    #[test]
    fn sun_is_below_the_horizon_at_local_midnight() {
        let unix = 1_710_936_000.0 + 12.0 * 3_600.0; // 2024-03-21 00:00 UTC (local midnight at lon 0)
        let p = solar_position(unix, 0.0, 0.0);
        assert!(p.elevation_rad < 0.0, "elevation {}°", p.elevation_rad.to_degrees());
    }

    #[test]
    fn winter_solstice_declination_is_about_minus_23_44_deg() {
        // Declination is internal; validate its visible consequence:
        // noon elevation at 51.5°N on the December solstice ≈ 90 − 51.5 − 23.44 ≈ 15°.
        let unix = 1_703_073_600.0; // 2023-12-21 12:00 UTC
        let p = solar_position(unix, 51.5 * DEG, -0.12 * DEG);
        let noon_el = p.elevation_rad.to_degrees();
        assert!(
            (12.0..19.0).contains(&noon_el),
            "expected ~15° noon elevation, got {noon_el}°"
        );
    }

    #[test]
    fn azimuth_is_a_full_circle() {
        let p = solar_position(1_710_951_600.0, 40.0 * DEG, -74.0 * DEG);
        assert!((0.0..std::f64::consts::TAU).contains(&p.azimuth_rad));
    }
}

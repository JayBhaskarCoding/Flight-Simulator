//! METAR decoder implementing the public ICAO WMO 401 report format.
//!
//! Grammar consumed (in report order):
//!
//! 1. `METAR` / `SPECI` designator
//! 2. `COR` / `AUTO` modifiers (before or after the station/time)
//! 3. station id (4 chars), `ddHHMMZ` observation time
//! 4. wind `dddssKT`, `dddssGggKT`, `VRBssKT`, optional `dddVddd`
//! 5. optional runway visual ranges `R24/1400V1600N`
//! 6. visibility: metres (`0400`, `9999`), `CAVOK`, or statute miles
//!    (`M1/4SM`, `2 1/2SM`)
//! 7. present weather (`-RA`, `+TSRA`, `BCFG`, `VCSH`, …)
//! 8. cloud groups (`FEW030`, `BKN045CB`, `VV003`, `CLR`, …)
//! 9. temperature/dewpoint `18/10`, `M03/M07`
//! 10. pressure `Q1013` / `A2992`
//! 11. recent weather `RE1056RA`, trend (`NOSIG`, `TEMPO …`) and `RMK`
//!     (remarks are consumed but not interpreted).

use std::fmt;

const KNOT_TO_MPS: f64 = 1852.0 / 3600.0;
const SM_TO_M: f64 = 1609.344;

/// Parse error with the offending token position.
#[derive(Debug, Clone, PartialEq)]
pub enum MetarError {
    /// Structural problem at group `index` of the raw report.
    Group { index: usize, message: String },
    /// A required group is missing.
    Missing { what: &'static str },
    /// A numeric field is malformed.
    Number { index: usize, token: String, what: &'static str },
}

impl fmt::Display for MetarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Group { index, message } => write!(f, "group {index}: {message}"),
            Self::Missing { what } => write!(f, "missing {what}"),
            Self::Number { index, token, what } => {
                write!(f, "group {index}: cannot parse {what} from '{token}'")
            }
        }
    }
}

impl std::error::Error for MetarError {}

type MetarResult<T> = Result<T, MetarError>;

/// Wind group.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wind {
    /// True direction the wind comes FROM, degrees (None for `VRB`).
    pub direction_deg: Option<f64>,
    /// Sustained speed, m/s.
    pub speed_mps: f64,
    /// Gust speed, m/s.
    pub gust_mps: Option<f64>,
    /// Variable-wind direction range, degrees.
    pub variable_deg: Option<(f64, f64)>,
}

impl Wind {
    /// Direction in degrees (0..360); for a `VRB` wind with a variable
    /// range returns the circular midpoint of that range.
    pub fn direction_or_variable_mid(&self) -> Option<f64> {
        self.direction_deg.or_else(|| {
            self.variable_deg.map(|(a, b)| {
                let delta = ((b - a) % 360.0 + 360.0) % 360.0;
                (a + delta / 2.0).rem_euclid(360.0)
            })
        })
    }
}

/// Runway visual range.
#[derive(Debug, Clone, PartialEq)]
pub struct Rvr {
    /// Runway designator, e.g. `24`.
    pub runway: String,
    /// Minimum visual range, m.
    pub min_m: f64,
    /// Maximum visual range, m (equal to min when not a range report).
    pub max_m: f64,
    /// True when the report uses `P`/`M` (beyond/unless threshold).
    pub beyond_threshold: Option<bool>,
}

/// Sky/obscuration condition of a cloud layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkyCondition {
    /// 1/8–2/8 cover.
    Few,
    /// 3/8–4/8 cover.
    Scattered,
    /// 5/8–7/8 cover.
    Broken,
    /// 8/8 cover.
    Overcast,
    /// `VV` vertical visibility (obscured sky).
    VerticalVisibility,
}

/// One cloud layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CloudLayer {
    /// Cover type.
    pub condition: SkyCondition,
    /// Layer base, feet above aerodrome level.
    pub base_ft: f64,
    /// `CB` tag.
    pub cumulonimbus: bool,
    /// `MAM` tag.
    pub mammatus: bool,
}

/// Intensity qualifier of a weather group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Intensity {
    /// No qualifier.
    #[default]
    Moderate,
    /// `-` prefix.
    Light,
    /// `+` prefix.
    Heavy,
}

/// Descriptor half of a weather group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Descriptor {
    /// `MI` shallow.
    Shallow,
    /// `PR` partial.
    Partial,
    /// `BC` patches.
    Patches,
    /// `DR` drifting.
    Drifting,
    /// `BL` blowing.
    Blowing,
    /// `SH` showers.
    Showers,
    /// `TS` thunderstorm.
    Thunderstorm,
    /// `FZ` freezing.
    Freezing,
}

/// Precipitation/obscuration phenomenon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phenomenon {
    /// `DZ` drizzle.
    Drizzle,
    /// `RA` rain.
    Rain,
    /// `SN` snow.
    Snow,
    /// `SG` snow grains.
    SnowGrains,
    /// `IC` ice crystals.
    IceCrystals,
    /// `PL` ice pellets.
    IcePellets,
    /// `GR` hail.
    Hail,
    /// `GS` small hail/snow pellets.
    SmallHail,
    /// `UP` unknown precipitation.
    UnknownPrecipitation,
    /// `FG` fog.
    Fog,
    /// `BR` mist.
    Mist,
    /// `DU` widespread dust.
    Dust,
    /// `SA` sand.
    Sand,
    /// `HZ` haze.
    Haze,
    /// `FU` smoke.
    Smoke,
    /// `VA` volcanic ash.
    VolcanicAsh,
    /// `PY` spray.
    Spray,
}

/// One present-weather group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeatherPhenomenon {
    /// Intensity qualifier.
    pub intensity: Intensity,
    /// Descriptor (`SH`, `TS`, `FZ`, …).
    pub descriptor: Option<Descriptor>,
    /// One or more phenomena (e.g. `RA` + `SN` in `RASN`).
    pub phenomena: Vec<Phenomenon>,
    /// `VC` — weather in the vicinity (not at the station).
    pub vicinity: bool,
}

/// Recent weather group (`RE1056RA`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecentWeather {
    /// Phenomenon observed.
    pub phenomenon: Phenomenon,
    /// Hour UTC of the recent observation.
    pub hour_utc: u8,
    /// Minute UTC.
    pub minute_utc: u8,
}

/// Decoded METAR report.
#[derive(Debug, Clone, PartialEq)]
pub struct Metar {
    /// `true` when this is a `SPECI` (special) report.
    pub special: bool,
    /// Issued as a correction.
    pub corrected: bool,
    /// Automatic station (no human observer).
    pub automatic: bool,
    /// ICAO station identifier.
    pub station: String,
    /// Day of month of the observation.
    pub day_utc: u8,
    /// Hour UTC.
    pub hour_utc: u8,
    /// Minute UTC.
    pub minute_utc: u8,
    /// Wind group.
    pub wind: Wind,
    /// Runway visual ranges.
    pub rvr: Vec<Rvr>,
    /// Horizontal visibility, metres.
    pub visibility_m: f64,
    /// `CAVOK` was reported (visibility ≥ 10 km, no cloud/weather).
    pub cavok: bool,
    /// Visibility below the reporting minimum (`M` prefix).
    pub visibility_below_minima: bool,
    /// Present weather.
    pub weather: Vec<WeatherPhenomenon>,
    /// Recent weather.
    pub recent: Vec<RecentWeather>,
    /// Cloud layers in ascending order (empty = clear or `CLR`/`NSC`).
    pub clouds: Vec<CloudLayer>,
    /// Air temperature, °C.
    pub temperature_c: i8,
    /// Dewpoint temperature, °C.
    pub dewpoint_c: i8,
    /// Altimeter setting, Pa (None when neither `Q` nor `A` given).
    pub pressure_pa: Option<f64>,
    /// Trend/forecast text (`NOSIG`, `TEMPO …`) retained verbatim.
    pub trend: String,
}

impl Metar {
    /// Decode a raw METAR string.
    pub fn parse(raw: &str) -> MetarResult<Self> {
        let tokens: Vec<&str> = raw.split_whitespace().collect();
        let mut i = 0usize;
        let mut gi = 0usize; // semantic group index for errors

        let mut special = false;
        let mut corrected = false;
        let mut automatic = false;

        // 1. designator
        match tokens.get(i).copied() {
            Some("METAR") => i += 1,
            Some("SPECI") => {
                special = true;
                i += 1;
            }
            _ => { /* designator may be stripped by the collector */ }
        }
        // 2. leading modifiers
        while let Some(&t) = tokens.get(i) {
            match t {
                "COR" | "CCA" | "CCB" => {
                    corrected = true;
                    i += 1;
                    gi += 1;
                }
                "AUTO" => {
                    automatic = true;
                    i += 1;
                    gi += 1;
                }
                _ => break,
            }
        }
        // 3. station
        let station = *tokens.get(i).ok_or(MetarError::Missing { what: "station" })?;
        if station.len() < 3 || station.len() > 5 || !station.chars().all(|c| c.is_ascii_alphanumeric()) || station.chars().any(|c| c.is_ascii_digit()) {
            return Err(MetarError::Group {
                index: gi,
                message: format!("'{station}' is not a station identifier"),
            });
        }
        i += 1;
        gi += 1;
        // 4. ddHHMMZ
        let time_tok = *tokens.get(i).ok_or(MetarError::Missing { what: "observation time" })?;
        let (day_utc, hour_utc, minute_utc) = parse_time(time_tok, gi)?;
        i += 1;
        gi += 1;
        // post-time modifiers
        while let Some(&t) = tokens.get(i) {
            match t {
                "COR" => {
                    corrected = true;
                    i += 1;
                    gi += 1;
                }
                "AUTO" => {
                    automatic = true;
                    i += 1;
                    gi += 1;
                }
                _ => break,
            }
        }

        // 5. wind (+ optional separate direction-variation group)
        let mut wind = parse_wind(tokens[i], gi)?;
        i += 1;
        gi += 1;
        if let Some(&t) = tokens.get(i) {
            if let Some((from, to)) = parse_variable_range(t, gi)? {
                wind.variable_deg = Some((from, to));
                i += 1;
                gi += 1;
            }
        }

        // 6. optional RVR groups
        let mut rvr = Vec::new();
        while let Some(&t) = tokens.get(i) {
            if t.starts_with('R') && t.len() > 1 && t.as_bytes()[1].is_ascii_digit() {
                rvr.push(parse_rvr(t, gi)?);
                i += 1;
                gi += 1;
            } else {
                break;
            }
        }

        // 7. visibility (may span two tokens for mixed numbers)
        let (visibility_m, cavok, below, consumed) = parse_visibility(&tokens[i..], gi)?;
        i += consumed;
        gi += 1;

        // 8. weather + clouds (interleaved), until temp/pressure/trend
        let mut weather: Vec<WeatherPhenomenon> = Vec::new();
        let mut recent: Vec<RecentWeather> = Vec::new();
        let mut clouds: Vec<CloudLayer> = Vec::new();
        let mut temperature_c: Option<i8> = None;
        let mut dewpoint_c: Option<i8> = None;
        let mut pressure_pa: Option<f64> = None;

        while let Some(&t) = tokens.get(i) {
            if cavok {
                // CAVOK replaces weather/cloud groups; parsing resumes at
                // temperature/pressure which follows immediately.
                if !is_temperature_token(t) && parse_pressure(t).is_none() {
                    break;
                }
            }
            if t == "RMK" {
                break;
            }
            if t == "NOSIG" || t == "TEMPO" || t == "BECMG" || t == "AMD" {
                break;
            }
            if t.starts_with("FM") && t.len() >= 6 && t[2..].chars().all(|c| c.is_ascii_digit()) {
                break;
            }
            if is_temperature_token(t) {
                let (temp, dew) = parse_temperatures(t, gi)?;
                temperature_c = Some(temp);
                dewpoint_c = Some(dew);
                i += 1;
                gi += 1;
                continue;
            }
            if let Some(p) = parse_pressure(t) {
                pressure_pa = Some(p);
                i += 1;
                gi += 1;
                continue;
            }
            if let Some(layer) = parse_cloud(t, gi)? {
                clouds.push(layer);
                i += 1;
                gi += 1;
                continue;
            }
            if t == "NSW" {
                weather.clear();
                i += 1;
                gi += 1;
                continue;
            }
            if t == "CLR" || t == "SKC" || t == "NSC" || t == "NCD" {
                i += 1;
                gi += 1;
                continue;
            }
            if let Some(r) = parse_recent(t, gi)? {
                recent.push(r);
                i += 1;
                gi += 1;
                continue;
            }
            if looks_like_weather(t) {
                weather.push(parse_weather(t, gi)?);
                i += 1;
                gi += 1;
                continue;
            }
            return Err(MetarError::Group {
                index: gi,
                message: format!("unexpected group '{t}'"),
            });
        }

        // 9. trend (everything before RMK)
        let mut trend = String::new();
        if let Some(&t) = tokens.get(i) {
            if t != "RMK" {
                trend = tokens[i..].join(" ");
            }
        }

        let temperature_c = temperature_c.ok_or(MetarError::Missing { what: "temperature" })?;
        let dewpoint_c = dewpoint_c.ok_or(MetarError::Missing { what: "dewpoint" })?;

        // CAVOK defaults
        let (visibility_m, clouds) = if cavok {
            (10_000.0, clouds)
        } else {
            (visibility_m, clouds)
        };

        Ok(Metar {
            special,
            corrected,
            automatic,
            station: station.to_string(),
            day_utc,
            hour_utc,
            minute_utc,
            wind,
            rvr,
            visibility_m,
            cavok,
            visibility_below_minima: below,
            weather,
            recent,
            clouds,
            temperature_c,
            dewpoint_c,
            pressure_pa,
            trend,
        })
    }

    /// Approximate ISA conditions implied by this report at the
    /// aerodrome elevation: returns `(conditions, relative humidity)`.
    ///
    /// The temperature offset is measured against the ISA temperature at
    /// `field_elevation_m`; pressure comes from the `Q`/`A` group; RH is
    /// derived from temperature/dewpoint (Buck saturation curve).
    pub fn isa_conditions(
        &self,
        field_elevation_m: f64,
    ) -> (crate::isa::IsaConditions, f64) {
        use crate::isa::{saturation_vapour_pressure_pa, IsaConditions, IsaModel};
        let model = IsaModel::standard();
        let isa_t = model.at_geometric_m(field_elevation_m).temperature_k;
        let measured_t = self.temperature_c as f64 + 273.15;
        let sea_level_pressure = self.pressure_pa.unwrap_or(crate::isa::P0);
        let e_dew = saturation_vapour_pressure_pa(self.dewpoint_c as f64 + 273.15);
        let e_sat = saturation_vapour_pressure_pa(self.temperature_c as f64 + 273.15);
        let rh = if e_sat > 0.0 { (e_dew / e_sat).clamp(0.0, 1.0) } else { 0.0 };
        (
            IsaConditions {
                delta_t_k: measured_t - isa_t,
                sea_level_pressure_pa: sea_level_pressure,
            },
            rh,
        )
    }

    /// Wind speed in m/s.
    pub fn wind_speed_mps(&self) -> f64 {
        self.wind.speed_mps
    }

    /// Gust speed in m/s (sustained speed when no gust reported).
    pub fn wind_gust_mps(&self) -> f64 {
        self.wind.gust_mps.unwrap_or(self.wind.speed_mps)
    }

    /// Lowest cloud base in feet (None when clear).
    pub fn lowest_cloud_base_ft(&self) -> Option<f64> {
        self.clouds.iter().map(|l| l.base_ft).fold(None, |a: Option<f64>, b| {
            Some(a.map_or(b, |v| v.min(b)))
        })
    }
}

// ---------------------------------------------------------------------------
// group parsers
// ---------------------------------------------------------------------------

fn parse_time(tok: &str, index: usize) -> MetarResult<(u8, u8, u8)> {
    let t = tok
        .strip_suffix('Z')
        .ok_or(MetarError::Group { index, message: format!("time '{tok}' must end with Z") })?;
    if t.len() != 6 || !t.chars().all(|c| c.is_ascii_digit()) {
        return Err(MetarError::Number { index, token: tok.to_string(), what: "ddHHMMZ time" });
    }
    let day: u8 = t[0..2].parse().unwrap();
    let hour: u8 = t[2..4].parse().unwrap();
    let minute: u8 = t[4..6].parse().unwrap();
    if day == 0 || day > 31 || hour > 23 || minute > 59 {
        return Err(MetarError::Number { index, token: tok.to_string(), what: "ddHHMMZ time range" });
    }
    Ok((day, hour, minute))
}

fn parse_wind(tok: &str, index: usize) -> MetarResult<Wind> {
    if !tok.ends_with("KT") {
        return Err(MetarError::Group { index, message: format!("wind '{tok}' must end with KT") });
    }
    let core = &tok[..tok.len() - 2];

    let (direction, speed_part) = if let Some(rest) = core.strip_prefix("VRB") {
        (None, rest)
    } else {
        if core.len() < 5 {
            return Err(MetarError::Number { index, token: tok.to_string(), what: "wind group" });
        }
        let d = &core[0..3];
        if !d.chars().all(|c| c.is_ascii_digit()) {
            return Err(MetarError::Number { index, token: tok.to_string(), what: "wind direction" });
        }
        let dir: f64 = d.parse().unwrap();
        if dir >= 360.0 && dir != 360.0 {
            return Err(MetarError::Number { index, token: tok.to_string(), what: "wind direction < 360" });
        }
        (Some(if dir == 360.0 { 0.0 } else { dir }), &core[3..])
    };

    // speed part: ### or ###G### — digits before optional G
    let (speed_str, gust_str) = match speed_part.find('G') {
        Some(pos) => (&speed_part[..pos], Some(&speed_part[pos + 1..])),
        None => (speed_part, None),
    };
    if speed_str.is_empty() || !speed_str.chars().all(|c| c.is_ascii_digit()) {
        return Err(MetarError::Number { index, token: tok.to_string(), what: "wind speed" });
    }
    let speed_kt: f64 = speed_str.parse().unwrap();
    let gust_mps = match gust_str {
        Some(g) if !g.is_empty() && g.chars().all(|c| c.is_ascii_digit()) => {
            Some(g.parse::<f64>().unwrap() * KNOT_TO_MPS)
        }
        Some(_) => {
            return Err(MetarError::Number { index, token: tok.to_string(), what: "gust speed" })
        }
        None => None,
    };

    Ok(Wind {
        direction_deg: direction,
        speed_mps: speed_kt * KNOT_TO_MPS,
        gust_mps,
        variable_deg: None,
    })
}

/// `dddVddd` direction-variation group (separate group after the wind).
fn parse_variable_range(tok: &str, index: usize) -> MetarResult<Option<(f64, f64)>> {
    if tok.len() != 7 || tok.as_bytes()[3] != b'V' {
        return Ok(None);
    }
    let (a, b) = (&tok[..3], &tok[4..]);
    if !a.chars().all(|c| c.is_ascii_digit()) || !b.chars().all(|c| c.is_ascii_digit()) {
        return Err(MetarError::Number {
            index,
            token: tok.to_string(),
            what: "wind direction variation",
        });
    }
    let from: f64 = a.parse().unwrap();
    let to: f64 = b.parse().unwrap();
    if from >= 360.0 || to >= 360.0 {
        return Err(MetarError::Number {
            index,
            token: tok.to_string(),
            what: "wind direction variation < 360",
        });
    }
    Ok(Some((from, to)))
}

fn parse_rvr(tok: &str, index: usize) -> MetarResult<Rvr> {
    // R24/1400 | R24/1400V1600 | RM24/M1600V1400N | P: prefix, N/U/D trend
    let body = &tok[1..];
    let slash = body.find('/').ok_or(MetarError::Group {
        index,
        message: format!("RVR '{tok}' missing '/'"),
    })?;
    let (runway, rest) = body.split_at(slash);
    let rest = &rest[1..];
    if runway.is_empty() || !runway.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(MetarError::Group { index, message: format!("bad runway '{runway}'") });
    }
    // strip trend tendency letter (N/U/D) if present
    let rest = rest.trim_end_matches(['N', 'U', 'D']);
    let (min_s, max_s) = match rest.find('V') {
        Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
        None => (rest, None),
    };
    let parse_len = |s: &str| -> MetarResult<(f64, bool)> {
        let beyond = if let Some(s) = s.strip_prefix('P') {
            (s, true)
        } else if let Some(s) = s.strip_prefix('M') {
            (s, false)
        } else {
            (s, false)
        };
        if beyond.0.is_empty() || !beyond.0.chars().all(|c| c.is_ascii_digit()) {
            return Err(MetarError::Number { index, token: tok.to_string(), what: "RVR value" });
        }
        let v: f64 = beyond.0.parse().unwrap();
        // RVR in metres unless suffixed FT (handled by caller unit choice)
        Ok((v, beyond.1))
    };
    let (min_v, min_beyond) = parse_len(min_s)?;
    let (max_v, max_beyond) = match max_s {
        Some(s) => parse_len(s)?,
        None => (min_v, min_beyond),
    };
    Ok(Rvr {
        runway: runway.to_string(),
        min_m: min_v,
        max_m: max_v,
        beyond_threshold: if min_beyond || max_beyond { Some(max_beyond) } else { None },
    })
}

/// Returns (metres, cavok, below_minima, tokens_consumed).
fn parse_visibility(tokens: &[&str], index: usize) -> MetarResult<(f64, bool, bool, usize)> {
    let tok = *tokens.first().ok_or(MetarError::Missing { what: "visibility" })?;
    if tok == "CAVOK" {
        return Ok((10_000.0, true, false, 1));
    }
    // mixed number: `<int> <frac>SM` spans two tokens
    if let Some(next) = tokens.get(1) {
        if next.ends_with("SM") && next.contains('/') {
            let whole = parse_sm_part(tok, index, tok)?;
            let frac = parse_sm_fraction(&next[..next.len() - 2], index, tok)?;
            let below = tok.starts_with('M');
            return Ok(((whole + frac) * SM_TO_M, false, below, 2));
        }
    }
    // statute miles: `4SM`, `1/2SM`, `M1/4SM`
    if let Some(body) = tok.strip_suffix("SM") {
        let (below, body) = match body.strip_prefix('M') {
            Some(rest) => (true, rest),
            None => (false, body),
        };
        let miles = if body.contains('/') {
            parse_sm_fraction(body, index, tok)?
        } else {
            parse_sm_part(body, index, tok)?
        };
        return Ok((miles * SM_TO_M, false, below, 1));
    }
    // metric visibility: exactly 4 digits, metres (9999 = >= 10 km)
    if tok.len() == 4 && tok.chars().all(|c| c.is_ascii_digit()) {
        let v: f64 = tok.parse().unwrap();
        let v = if v == 9999.0 { 10_000.0 } else { v };
        return Ok((v, false, false, 1));
    }
    Err(MetarError::Group { index, message: format!("cannot parse visibility '{tok}'") })
}

fn parse_sm_part(s: &str, index: usize, tok: &str) -> MetarResult<f64> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) {
        return Err(MetarError::Number { index, token: tok.to_string(), what: "statute miles" });
    }
    s.parse()
        .map_err(|_| MetarError::Number { index, token: tok.to_string(), what: "statute miles" })
}

fn parse_sm_fraction(s: &str, index: usize, tok: &str) -> MetarResult<f64> {
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() != 2 {
        return Err(MetarError::Group { index, message: format!("bad visibility fraction '{s}'") });
    }
    let n: f64 = parts[0]
        .parse()
        .map_err(|_| MetarError::Number { index, token: tok.to_string(), what: "fraction" })?;
    let d: f64 = parts[1]
        .parse()
        .map_err(|_| MetarError::Number { index, token: tok.to_string(), what: "fraction" })?;
    if d == 0.0 {
        return Err(MetarError::Group { index, message: "zero fraction denominator".into() });
    }
    Ok(n / d)
}

fn is_temperature_token(t: &str) -> bool {
    if !t.contains('/') {
        return false;
    }
    let parts: Vec<&str> = t.split('/').collect();
    parts.len() == 2 && parts.iter().all(|p| {
        let p = p.strip_prefix('M').unwrap_or(p);
        !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()) && p.len() <= 2
    })
}

fn parse_temperatures(t: &str, index: usize) -> MetarResult<(i8, i8)> {
    let parse1 = |s: &str| -> MetarResult<i8> {
        let neg = s.starts_with('M');
        let digits = s.strip_prefix('M').unwrap_or(s);
        let v: i8 = digits
            .parse()
            .map_err(|_| MetarError::Number { index, token: t.to_string(), what: "temperature" })?;
        Ok(if neg { -v } else { v })
    };
    let parts: Vec<&str> = t.split('/').collect();
    if parts.len() != 2 {
        return Err(MetarError::Group { index, message: format!("bad temperatures '{t}'") });
    }
    Ok((parse1(parts[0])?, parse1(parts[1])?))
}

fn parse_pressure(t: &str) -> Option<f64> {
    if t.len() != 5 {
        return None;
    }
    let digits = &t[1..];
    if !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let v: f64 = digits.parse().ok()?;
    match &t[0..1] {
        "Q" => Some(v * 100.0),            // hPa → Pa
        "A" => Some(v / 100.0 * 3386.389), // inHg (×0.01) → Pa
        _ => None,
    }
}

fn parse_cloud(t: &str, index: usize) -> MetarResult<Option<CloudLayer>> {
    // Non-numeric groups (CLR/SKC/NSC/NCD/…) are not layers.
    if !t.chars().any(|c| c.is_ascii_digit()) {
        return Ok(None);
    }
    // Strip optional cloud-type tags.
    let (main, cumulonimbus, mammatus) = if let Some(m) = t.strip_suffix("CB") {
        (m, true, false)
    } else if let Some(m) = t.strip_suffix("MAM") {
        (m, false, true)
    } else {
        (t, false, false)
    };
    let split = main
        .char_indices()
        .find(|(_, c)| c.is_ascii_digit())
        .map(|(i, _)| i)
        .unwrap_or(0);
    let (code, rest) = main.split_at(split);
    let condition = match code {
        "FEW" => SkyCondition::Few,
        "SCT" => SkyCondition::Scattered,
        "BKN" => SkyCondition::Broken,
        "OVC" => SkyCondition::Overcast,
        "VV" => SkyCondition::VerticalVisibility,
        _ => return Ok(None),
    };
    if rest.len() != 3 || !rest.chars().all(|c| c.is_ascii_digit()) {
        return Err(MetarError::Number { index, token: t.to_string(), what: "cloud base height" });
    }
    let base_ft: f64 = rest.parse().unwrap();
    Ok(Some(CloudLayer {
        condition,
        base_ft: base_ft * 100.0,
        cumulonimbus,
        mammatus,
    }))
}

fn looks_like_weather(t: &str) -> bool {
    if t.is_empty() {
        return false;
    }
    let body = t.strip_prefix('-').or_else(|| t.strip_prefix('+')).unwrap_or(t);
    let body = body.strip_prefix("VC").unwrap_or(body);
    if body.is_empty() {
        return false;
    }
    // descriptors/phenomena are always 2-letter pairs
    body.len() % 2 == 0
        && body.chars().all(|c| c.is_ascii_uppercase())
        && WEATHER_TOKENS.iter().any(|w| body.contains(w))
}

const WEATHER_TOKENS: &[&str] = &[
    "MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ", "DZ", "RA", "SN", "SG", "IC", "PL", "GR",
    "GS", "UP", "FG", "BR", "DU", "SA", "HZ", "FU", "VA", "PY",
];

fn descriptor_of(pair: &str) -> Option<Descriptor> {
    Some(match pair {
        "MI" => Descriptor::Shallow,
        "PR" => Descriptor::Partial,
        "BC" => Descriptor::Patches,
        "DR" => Descriptor::Drifting,
        "BL" => Descriptor::Blowing,
        "SH" => Descriptor::Showers,
        "TS" => Descriptor::Thunderstorm,
        "FZ" => Descriptor::Freezing,
        _ => return None,
    })
}

fn phenomenon_of(pair: &str) -> Option<Phenomenon> {
    Some(match pair {
        "DZ" => Phenomenon::Drizzle,
        "RA" => Phenomenon::Rain,
        "SN" => Phenomenon::Snow,
        "SG" => Phenomenon::SnowGrains,
        "IC" => Phenomenon::IceCrystals,
        "PL" => Phenomenon::IcePellets,
        "GR" => Phenomenon::Hail,
        "GS" => Phenomenon::SmallHail,
        "UP" => Phenomenon::UnknownPrecipitation,
        "FG" => Phenomenon::Fog,
        "BR" => Phenomenon::Mist,
        "DU" => Phenomenon::Dust,
        "SA" => Phenomenon::Sand,
        "HZ" => Phenomenon::Haze,
        "FU" => Phenomenon::Smoke,
        "VA" => Phenomenon::VolcanicAsh,
        "PY" => Phenomenon::Spray,
        _ => return None,
    })
}

fn parse_weather(t: &str, index: usize) -> MetarResult<WeatherPhenomenon> {
    let mut s = t;
    let intensity = if let Some(rest) = s.strip_prefix('-') {
        s = rest;
        Intensity::Light
    } else if let Some(rest) = s.strip_prefix('+') {
        s = rest;
        Intensity::Heavy
    } else {
        Intensity::Moderate
    };
    let vicinity = if let Some(rest) = s.strip_prefix("VC") {
        s = rest;
        true
    } else {
        false
    };
    // optional descriptor (first pair)
    let mut descriptor = None;
    if s.len() >= 4 {
        if let Some(d) = descriptor_of(&s[..2]) {
            descriptor = Some(d);
            s = &s[2..];
        }
    } else if s.len() == 2 {
        // bare descriptor (e.g. "SH" alone) or single phenomenon
        if let Some(d) = descriptor_of(s) {
            // a lone descriptor without phenomenon is only valid for TS/SH
            // in practice reports always follow with a phenomenon; treat
            // bare SH/TS as descriptor-only groups
            return Ok(WeatherPhenomenon {
                intensity,
                descriptor: Some(d),
                phenomena: Vec::new(),
                vicinity,
            });
        }
    }
    let mut phenomena = Vec::new();
    let bytes: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i + 1 < bytes.len() || (i < bytes.len() && bytes.len() % 2 == 0) {
        if i + 1 >= bytes.len() {
            break;
        }
        let pair: String = bytes[i..i + 2].iter().collect();
        match phenomenon_of(&pair) {
            Some(p) => phenomena.push(p),
            None => {
                return Err(MetarError::Group {
                    index,
                    message: format!("unknown weather element '{pair}' in '{t}'"),
                })
            }
        }
        i += 2;
    }
    if phenomena.is_empty() && descriptor.is_none() && !vicinity {
        return Err(MetarError::Group { index, message: format!("empty weather group '{t}'") });
    }
    Ok(WeatherPhenomenon { intensity, descriptor, phenomena, vicinity })
}

fn parse_recent(t: &str, index: usize) -> MetarResult<Option<RecentWeather>> {
    if !t.starts_with("RE") || t.len() < 6 {
        return Ok(None);
    }
    let body = &t[2..];
    if body.len() < 6 {
        return Ok(None);
    }
    let (time, phen) = body.split_at(4);
    if time.len() != 4 || !time.chars().all(|c| c.is_ascii_digit()) {
        return Ok(None);
    }
    if phen.len() % 2 != 0 || phen.is_empty() {
        return Ok(None);
    }
    let hour: u8 = time[0..2]
        .parse()
        .map_err(|_| MetarError::Number { index, token: t.to_string(), what: "recent weather time" })?;
    let minute: u8 = time[2..4]
        .parse()
        .map_err(|_| MetarError::Number { index, token: t.to_string(), what: "recent weather time" })?;
    if hour > 23 || minute > 59 {
        return Err(MetarError::Number { index, token: t.to_string(), what: "recent weather time range" });
    }
    let p = phenomenon_of(&phen[..2]).ok_or(MetarError::Group {
        index,
        message: format!("unknown recent phenomenon '{phen}'"),
    })?;
    Ok(Some(RecentWeather { phenomenon: p, hour_utc: hour, minute_utc: minute }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_full_report() {
        let raw = "METAR EDDF 121050Z 24015G28KT 9999 -RA SCT040 BKN070CB 18/10 Q1012";
        let m = Metar::parse(raw).unwrap();
        assert_eq!(m.station, "EDDF");
        assert!(!m.special);
        assert_eq!((m.day_utc, m.hour_utc, m.minute_utc), (12, 10, 50));
        assert_eq!(m.wind.direction_deg, Some(240.0));
        assert!((m.wind.speed_mps - 15.0 * KNOT_TO_MPS).abs() < 1e-9);
        assert!((m.wind_gust_mps() - 28.0 * KNOT_TO_MPS).abs() < 1e-9);
        assert!((m.visibility_m - 10_000.0).abs() < 1e-9); // 9999 = >=10 km
        assert_eq!(m.weather.len(), 1);
        assert_eq!(m.weather[0].intensity, Intensity::Light);
        assert_eq!(m.weather[0].phenomena, vec![Phenomenon::Rain]);
        assert_eq!(m.clouds.len(), 2);
        assert_eq!(m.clouds[0].condition, SkyCondition::Scattered);
        assert_eq!(m.clouds[0].base_ft, 4000.0);
        assert_eq!(m.clouds[1].condition, SkyCondition::Broken);
        assert_eq!(m.temperature_c, 18);
        assert_eq!(m.dewpoint_c, 10);
        assert_eq!(m.pressure_pa, Some(101_200.0));
    }

    #[test]
    fn parses_caovok_and_speci() {
        let m = Metar::parse("SPECI EGLL 121050Z 18012KT CAVOK 15/09 Q1018 NOSIG").unwrap();
        assert!(m.special);
        assert!(m.cavok);
        assert!((m.visibility_m - 10_000.0).abs() < 1e-9);
        assert!(m.clouds.is_empty());
        assert_eq!(m.trend, "NOSIG");
        assert_eq!(m.pressure_pa, Some(101_800.0));
    }

    #[test]
    fn parses_statute_mile_visibility() {
        let m = Metar::parse(
            "METAR KJFK 121151Z 09008KT 1 1/2SM BR OVC007 12/11 A2989 RMK AO2 SLP118",
        )
        .unwrap();
        assert!((m.visibility_m - 1.5 * SM_TO_M).abs() < 1e-6, "{}", m.visibility_m);
        assert_eq!(m.weather[0].phenomena, vec![Phenomenon::Mist]);
        assert_eq!(m.clouds[0].base_ft, 700.0);
        // A2989 = 29.89 inHg
        let pa = m.pressure_pa.unwrap();
        assert!((pa - 29.89 * 3386.389).abs() < 0.5, "{pa}");

        let m2 = Metar::parse("METAR KSFO 121156Z 28012KT 1/4SM FG VV002 10/10 A2991").unwrap();
        assert!((m2.visibility_m - 0.25 * SM_TO_M).abs() < 1e-6);

        let m3 = Metar::parse("METAR KLAX 121151Z 07005KT M1/4SM OVC003 09/09 A2995").unwrap();
        assert!(m3.visibility_below_minima);
    }

    #[test]
    fn parses_variable_wind_and_gusts() {
        let m = Metar::parse("METAR AAAA 121000Z VRB04KT 180V240 9999 SCT030 12/08 Q1013").unwrap();
        assert_eq!(m.wind.direction_deg, None);
        assert_eq!(m.wind.variable_deg, Some((180.0, 240.0)));
        assert!((m.wind.speed_mps - 4.0 * KNOT_TO_MPS).abs() < 1e-9);

        let m2 = Metar::parse("METAR BBBB 121000Z 36005KT 9999 CLR 12/08 Q1013").unwrap();
        assert_eq!(m2.wind.direction_deg, Some(0.0)); // 360 normalised
        assert!(m2.clouds.is_empty());
    }

    #[test]
    fn parses_complex_weather_groups() {
        let m = Metar::parse(
            "METAR CCCC 121000Z 27010KT 2000 -SN BKN012 M02/M04 Q0998",
        )
        .unwrap();
        let w = &m.weather[0];
        assert_eq!(w.intensity, Intensity::Light);
        assert_eq!(w.descriptor, None);
        assert_eq!(w.phenomena, vec![Phenomenon::Snow]);

        let m2 = Metar::parse(
            "METAR DDDD 121000Z 27010KT 4000 +TSRA SCT020CB 15/11 Q1010",
        )
        .unwrap();
        let w2 = &m2.weather[0];
        assert_eq!(w2.intensity, Intensity::Heavy);
        assert_eq!(w2.descriptor, Some(Descriptor::Thunderstorm));
        assert_eq!(w2.phenomena, vec![Phenomenon::Rain]);
        assert!(m2.clouds[0].cumulonimbus);

        let m3 = Metar::parse("METAR EEEE 121000Z 00000KT 9999 VCSH BKN030 20/12 Q1012").unwrap();
        assert!(m3.weather[0].vicinity);
        assert_eq!(m3.weather[0].descriptor, Some(Descriptor::Showers));

        let m4 = Metar::parse("METAR FFFF 121000Z 10006KT 2000 BCFG M01/M03 Q1016").unwrap();
        assert_eq!(m4.weather[0].descriptor, Some(Descriptor::Patches));
        assert_eq!(m4.weather[0].phenomena, vec![Phenomenon::Fog]);

        let m5 = Metar::parse("METAR GGGG 121000Z 10006KT 6000 -FZDZ OVC004 00/M00 Q1017").unwrap();
        assert_eq!(m5.weather[0].descriptor, Some(Descriptor::Freezing));
        assert_eq!(m5.weather[0].phenomena, vec![Phenomenon::Drizzle]);
    }

    #[test]
    fn parses_negative_temperatures() {
        let m = Metar::parse("METAR HHHH 121000Z 30008KT 9999 NSC M03/M07 Q1022").unwrap();
        assert_eq!(m.temperature_c, -3);
        assert_eq!(m.dewpoint_c, -7);
        assert!(m.clouds.is_empty());
    }

    #[test]
    fn parses_rvr_and_recent_weather() {
        let m = Metar::parse(
            "METAR IIII 121000Z 05003KT R24/1200V1600N 0350 FG VV001 01/01 Q1030 RE0945FG",
        )
        .unwrap();
        assert_eq!(m.rvr.len(), 1);
        assert_eq!(m.rvr[0].runway, "24");
        assert_eq!((m.rvr[0].min_m, m.rvr[0].max_m), (1200.0, 1600.0));
        assert!((m.visibility_m - 350.0).abs() < 1e-9);
        assert_eq!(m.recent.len(), 1);
        assert_eq!(m.recent[0].phenomenon, Phenomenon::Fog);
        assert_eq!((m.recent[0].hour_utc, m.recent[0].minute_utc), (9, 45));
    }

    #[test]
    fn parses_auto_and_correction() {
        let m = Metar::parse("METAR AUTO KAAA 121000Z 18005KT 9999 FEW030 14/09 Q1015").unwrap();
        assert!(m.automatic);
        assert_eq!(m.station, "KAAA");

        let m2 = Metar::parse("METAR COR KBBB 121000Z 18005KT 9999 FEW030 14/09 Q1015").unwrap();
        assert!(m2.corrected);

        let m3 = Metar::parse("METAR KCCC 121000Z COR 18005KT 9999 FEW030 14/09 Q1015").unwrap();
        assert!(m3.corrected);
        assert_eq!(m3.station, "KCCC");
    }

    #[test]
    fn errors_are_specific() {
        let e = Metar::parse("METAR 121000Z 18005KT 9999 14/09 Q1015").unwrap_err();
        assert!(e.to_string().contains("station"), "{e}");

        let e = Metar::parse("METAR EDDF 121000 18005KT 9999 14/09 Q1015").unwrap_err();
        assert!(e.to_string().contains("time"), "{e}");

        let e = Metar::parse("METAR EDDF 121050Z 24015KT 9999 XY040 18/10 Q1012").unwrap_err();
        assert!(e.to_string().contains("XY040"), "{e}");

        // pressure is optional (ship reports): temperature/dewpoint required
        let m = Metar::parse("METAR EDDF 121050Z 24015KT 9999 18/10").unwrap();
        assert_eq!(m.pressure_pa, None);
        let e = Metar::parse("METAR EDDF 121050Z 24015KT 9999 Q1012").unwrap_err();
        assert!(e.to_string().contains("temperature"), "{e}");
    }

    #[test]
    fn isa_hints_match_report() {
        let m = Metar::parse("METAR LOWI 121050Z 27005KT 9999 SCT080 22/08 Q1016").unwrap();
        let (cond, rh) = m.isa_conditions(600.0);
        // ISA temp at 600 m ≈ 11.1 °C (284.25 K); reported 22 °C
        // → delta ≈ +10.9 K
        assert!((cond.delta_t_k - 10.9).abs() < 1.5, "{}", cond.delta_t_k);
        assert_eq!(cond.sea_level_pressure_pa, 101_600.0);
        // dewpoint 8 °C vs 22 °C: RH ≈ 41 %
        assert!(rh > 0.35 && rh < 0.55, "{rh}");
    }

    #[test]
    fn lowest_cloud_helper() {
        let m = Metar::parse("METAR ZZZZ 121000Z 18005KT 9999 SCT050 BKN080 14/09 Q1015").unwrap();
        assert_eq!(m.lowest_cloud_base_ft(), Some(5000.0));
    }
}

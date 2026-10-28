//! OpenStreetMap vectors: a dependency-free XML pull parser and the
//! node/way/tag model with road and building extraction.
//!
//! Scope (documented, deliberate): `node` and `way` elements are parsed;
//! `relation` subtrees are skipped wholesale — routing relations are not
//! needed for terrain roads or building footprints. Node-level `<tag>`
//! children are ignored the same way. Input is one tile's OSM XML
//! fragment (see [`crate::source::FileTileSource`]), which keeps memory
//! bounded by the tile budget rather than by an entire country export.
//!
//! The parser is a hand-written pull parser over `&str` (byte-positioned,
//! allocation-free until an entity or a consumer forces one). There is no
//! XML library dependency by design: the engine vendors only what it
//! needs, and OSM fragments use a tiny subset of XML.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

/// Errors from parsing or resolving OSM data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsmError {
    /// Malformed XML: `message` with the byte `offset` where it was found.
    Xml {
        /// Human-readable description.
        message: String,
        /// Byte offset into the input.
        offset: usize,
    },
    /// A required attribute was absent on an element.
    MissingAttr {
        /// Element name (`"node"`, `"way"`, ...).
        element: &'static str,
        /// Attribute name.
        attr: &'static str,
    },
    /// An attribute value failed to parse.
    Number {
        /// The offending raw value.
        value: String,
        /// What the value was supposed to be (`"node id"`, ...).
        what: &'static str,
    },
    /// A `<nd ref>` points at a node that is not in the fragment.
    MissingRef(i64),
}

impl fmt::Display for OsmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OsmError::Xml { message, offset } => {
                write!(f, "xml error at byte {offset}: {message}")
            }
            OsmError::MissingAttr { element, attr } => {
                write!(f, "missing attribute `{attr}` on `<{element}>`")
            }
            OsmError::Number { value, what } => write!(f, "invalid {what}: `{value}`"),
            OsmError::MissingRef(id) => write!(f, "way references missing node {id}"),
        }
    }
}

impl std::error::Error for OsmError {}

// ---------------------------------------------------------------------------
// XML pull parser
// ---------------------------------------------------------------------------

/// One XML event. Names and values borrow from the input unless an entity
/// reference forced an unescape (then they are owned).
#[derive(Debug, Clone, PartialEq)]
pub enum XmlEvent<'a> {
    /// `<name attr="value" ...>` (self-closing tags yield Start then End).
    Start {
        /// Element name.
        name: std::borrow::Cow<'a, str>,
        /// Attributes in document order.
        attrs: Vec<(std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>)>,
    },
    /// `</name>`.
    End(std::borrow::Cow<'a, str>),
    /// Character data between tags.
    Text(std::borrow::Cow<'a, str>),
}

/// Pull parser over a UTF-8 string.
#[derive(Debug, Clone)]
pub struct XmlParser<'a> {
    src: &'a str,
    pos: usize,
    /// Name of an element whose `/>` was already turned into a pending End.
    pending_end: Option<std::borrow::Cow<'a, str>>,
}

fn is_name_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b':' || b == b'.'
}

impl<'a> XmlParser<'a> {
    /// Create a parser; a leading UTF-8 BOM is skipped.
    pub fn new(src: &'a str) -> Self {
        let src = src.strip_prefix('\u{feff}').unwrap_or(src);
        XmlParser {
            src,
            pos: 0,
            pending_end: None,
        }
    }

    /// Current byte offset (for error messages).
    pub fn offset(&self) -> usize {
        self.pos
    }

    /// Next event; `None` at end of input.
    pub fn next_event(&mut self) -> Option<Result<XmlEvent<'a>, OsmError>> {
        if let Some(name) = self.pending_end.take() {
            return Some(Ok(XmlEvent::End(name)));
        }
        loop {
            let bytes = self.src.as_bytes();
            if self.pos >= bytes.len() {
                return None;
            }
            if bytes[self.pos] != b'<' {
                return Some(self.parse_text());
            }
            // At '<': swallow non-element markup and re-classify, because
            // a skip can land on whitespace before the next real tag.
            if bytes[self.pos..].starts_with(b"<?") {
                match self.src[self.pos..].find("?>") {
                    Some(k) => {
                        self.pos += k + 2;
                        continue;
                    }
                    None => {
                        return Some(self.err("unterminated processing instruction"));
                    }
                }
            }
            if bytes[self.pos..].starts_with(b"<!--") {
                match self.src[self.pos..].find("-->") {
                    Some(k) => {
                        self.pos += k + 3;
                        continue;
                    }
                    None => return Some(self.err("unterminated comment")),
                }
            }
            if bytes[self.pos..].starts_with(b"<!DOCTYPE")
                || bytes[self.pos..].starts_with(b"<!doctype")
                || bytes[self.pos..].starts_with(b"<!")
            {
                match self.src[self.pos..].find('>') {
                    Some(k) => {
                        self.pos += k + 1;
                        continue;
                    }
                    None => return Some(self.err("unterminated declaration")),
                }
            }
            break; // a real start/end tag
        }
        Some(self.parse_tag())
    }

    fn err<T>(&self, message: impl Into<String>) -> Result<T, OsmError> {
        Err(OsmError::Xml {
            message: message.into(),
            offset: self.pos,
        })
    }

    fn parse_text(&mut self) -> Result<XmlEvent<'a>, OsmError> {
        let start = self.pos;
        let bytes = self.src.as_bytes();
        while self.pos < bytes.len() && bytes[self.pos] != b'<' {
            self.pos += 1;
        }
        let raw = &self.src[start..self.pos];
        Ok(XmlEvent::Text(unescape(raw, start)?))
    }

    fn parse_tag(&mut self) -> Result<XmlEvent<'a>, OsmError> {
        let bytes = self.src.as_bytes();
        let start = self.pos;
        if bytes[start..].starts_with(b"<![CDATA[") {
            match self.src[start + 9..].find("]]>") {
                Some(k) => {
                    let text_start = start + 9;
                    let text = &self.src[text_start..text_start + k];
                    self.pos = text_start + k + 3;
                    return Ok(XmlEvent::Text(std::borrow::Cow::Borrowed(text)));
                }
                None => return self.err("unterminated CDATA section"),
            }
        }
        if bytes[start + 1..].starts_with(b"/") {
            // End tag.
            self.pos = start + 2;
            let name_start = self.pos;
            while self.pos < bytes.len() && bytes[self.pos] != b'>' {
                self.pos += 1;
            }
            if self.pos >= bytes.len() {
                self.pos = start;
                return self.err("unterminated end tag");
            }
            let name = self.src[name_start..self.pos].trim_end();
            self.pos += 1;
            return Ok(XmlEvent::End(std::borrow::Cow::Borrowed(name)));
        }

        // Start tag.
        self.pos = start + 1;
        let name_start = self.pos;
        while self.pos < bytes.len() && is_name_byte(bytes[self.pos]) {
            self.pos += 1;
        }
        if self.pos == name_start {
            self.pos = start;
            return self.err("element name missing after '<'");
        }
        let name = std::borrow::Cow::Borrowed(&self.src[name_start..self.pos]);

        let mut attrs: Vec<(std::borrow::Cow<'a, str>, std::borrow::Cow<'a, str>)> = Vec::new();
        loop {
            // Skip whitespace between attributes.
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= bytes.len() {
                self.pos = start;
                return self.err("unterminated start tag");
            }
            match bytes[self.pos] {
                b'>' => {
                    self.pos += 1;
                    return Ok(XmlEvent::Start { name, attrs });
                }
                b'/' => {
                    if self.pos + 1 < bytes.len() && bytes[self.pos + 1] == b'>' {
                        self.pos += 2;
                        self.pending_end = Some(name.clone());
                        return Ok(XmlEvent::Start { name, attrs });
                    }
                    self.pos = start;
                    return self.err("expected '/>' to close the element");
                }
                _ => {}
            }
            // Attribute name.
            let an_start = self.pos;
            while self.pos < bytes.len() && is_name_byte(bytes[self.pos]) {
                self.pos += 1;
            }
            if self.pos == an_start {
                self.pos = start;
                return self.err("malformed attribute name");
            }
            let aname = std::borrow::Cow::Borrowed(&self.src[an_start..self.pos]);
            // Optional whitespace, '='.
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= bytes.len() || bytes[self.pos] != b'=' {
                self.pos = start;
                return self.err(format!("expected '=' after attribute `{aname}`"));
            }
            self.pos += 1;
            while self.pos < bytes.len() && bytes[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos >= bytes.len() {
                self.pos = start;
                return self.err(format!("unterminated value for attribute `{aname}`"));
            }
            let quote = bytes[self.pos];
            if quote != b'"' && quote != b'\'' {
                self.pos = start;
                return self.err(format!("attribute `{aname}` value is not quoted"));
            }
            self.pos += 1;
            let vstart = self.pos;
            while self.pos < bytes.len() && bytes[self.pos] != quote {
                self.pos += 1;
            }
            if self.pos >= bytes.len() {
                self.pos = start;
                return self.err(format!("unterminated quote for attribute `{aname}`"));
            }
            let raw = &self.src[vstart..self.pos];
            self.pos += 1; // closing quote
            let value = unescape(raw, vstart)?;
            attrs.push((aname, value));
        }
    }
}

/// Resolve `&amp;`-style references; returns a borrowed slice when there
/// is nothing to expand.
fn unescape(raw: &str, base: usize) -> Result<std::borrow::Cow<'_, str>, OsmError> {
    if !raw.contains('&') {
        return Ok(std::borrow::Cow::Borrowed(raw));
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        let semi = tail.find(';').filter(|&k| k <= 10).ok_or_else(|| OsmError::Xml {
            message: "unterminated entity reference".into(),
            offset: base + amp,
        })?;
        let entity = &tail[1..semi];
        match entity {
            "amp" => out.push('&'),
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            _ => {
                let cp = if let Some(hex) = entity.strip_prefix("#x").or_else(|| entity.strip_prefix("#X"))
                {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(dec) = entity.strip_prefix('#') {
                    dec.parse::<u32>().ok()
                } else {
                    None
                };
                let ch = cp.and_then(char::from_u32).ok_or_else(|| OsmError::Xml {
                    message: format!("unknown or invalid entity `&{entity};`"),
                    offset: base + amp,
                })?;
                out.push(ch);
            }
        }
        rest = &tail[semi + 1..];
    }
    out.push_str(rest);
    Ok(std::borrow::Cow::Owned(out))
}

// ---------------------------------------------------------------------------
// OSM model
// ---------------------------------------------------------------------------

/// A mapped point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OsmNode {
    /// OSM node id.
    pub id: i64,
    /// Latitude (degrees).
    pub lat: f64,
    /// Longitude (degrees).
    pub lon: f64,
}

/// A way (polyline / polygon) with its tags.
#[derive(Debug, Clone, PartialEq)]
pub struct OsmWay {
    /// OSM way id.
    pub id: i64,
    /// Node references in order.
    pub refs: Vec<i64>,
    /// Tags, sorted by key (deterministic iteration).
    pub tags: BTreeMap<String, String>,
}

impl OsmWay {
    /// Tag value, if present.
    pub fn tag(&self, key: &str) -> Option<&str> {
        self.tags.get(key).map(String::as_str)
    }
}

/// Fragment bounds (from a `<bounds>` element).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OsmBounds {
    /// Southern latitude.
    pub min_lat: f64,
    /// Western longitude.
    pub min_lon: f64,
    /// Northern latitude.
    pub max_lat: f64,
    /// Eastern longitude.
    pub max_lon: f64,
}

/// Parsed fragment: nodes, ways and optional bounds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OsmMap {
    /// All nodes by id.
    pub nodes: HashMap<i64, OsmNode>,
    /// All ways in document order.
    pub ways: Vec<OsmWay>,
    /// Fragment bounds, when a complete `<bounds>` element was present.
    pub bounds: Option<OsmBounds>,
}

impl OsmMap {
    /// Number of nodes and ways (used by tests and stats).
    pub fn counts(&self) -> (usize, usize) {
        (self.nodes.len(), self.ways.len())
    }

    /// Resolve a way's node references into `(lat, lon)` points.
    pub fn resolve_points(&self, way: &OsmWay) -> Result<Vec<(f64, f64)>, OsmError> {
        let mut pts = Vec::with_capacity(way.refs.len());
        for r in &way.refs {
            let n = self.nodes.get(r).ok_or(OsmError::MissingRef(*r))?;
            pts.push((n.lat, n.lon));
        }
        Ok(pts)
    }
}

/// Parse an OSM XML fragment (`&str`).
pub fn parse_osm(input: &str) -> Result<OsmMap, OsmError> {
    let mut map = OsmMap::default();
    let mut parser = XmlParser::new(input);
    let mut cur_way: Option<OsmWay> = None;
    // Skip depth for out-of-scope subtrees (relations).
    let mut skip_depth: usize = 0;

    while let Some(ev) = parser.next_event() {
        match ev? {
            XmlEvent::Start { name, attrs } => {
                if skip_depth > 0 {
                    skip_depth += 1;
                    continue;
                }
                match &*name {
                    "node" => {
                        let id = req_attr_i64(&attrs, "node", "id")?;
                        let lat = req_attr_f64(&attrs, "node", "lat")?;
                        let lon = req_attr_f64(&attrs, "node", "lon")?;
                        map.nodes.insert(
                            id,
                            OsmNode {
                                id,
                                lat,
                                lon,
                            },
                        );
                    }
                    "way" => {
                        if cur_way.is_some() {
                            return Err(OsmError::Xml {
                                message: "nested <way> elements".into(),
                                offset: parser.offset(),
                            });
                        }
                        let id = req_attr_i64(&attrs, "way", "id")?;
                        cur_way = Some(OsmWay {
                            id,
                            refs: Vec::new(),
                            tags: BTreeMap::new(),
                        });
                    }
                    "relation" => {
                        skip_depth = 1;
                    }
                    "nd" => {
                        let way = cur_way.as_mut().ok_or_else(|| OsmError::Xml {
                            message: "<nd> outside of a <way>".into(),
                            offset: parser.offset(),
                        })?;
                        way.refs.push(req_attr_i64(&attrs, "nd", "ref")?);
                    }
                    "tag" => {
                        if let Some(way) = cur_way.as_mut() {
                            let k = req_attr(&attrs, "tag", "k")?;
                            let v = req_attr(&attrs, "tag", "v")?;
                            way.tags.insert(k.to_string(), v.to_string());
                        }
                        // Node-level tags are out of scope and ignored.
                    }
                    "bounds" => {
                        if map.bounds.is_some() {
                            return Err(OsmError::Xml {
                                message: "duplicate <bounds> element".into(),
                                offset: parser.offset(),
                            });
                        }
                        map.bounds = Some(parse_bounds(&attrs)?);
                    }
                    _ => {
                        // Unknown element (osm, metadata, ...): its End is
                        // ignored below; children with known names would be
                        // misread, so unknown *containers* are skipped too
                        // only when they can hold content we care about —
                        // in practice only <relation> does, handled above.
                    }
                }
            }
            XmlEvent::End(name) => {
                if skip_depth > 0 {
                    skip_depth -= 1;
                    continue;
                }
                if &*name == "way" {
                    let way = cur_way.take().ok_or_else(|| OsmError::Xml {
                        message: "stray </way>".into(),
                        offset: parser.offset(),
                    })?;
                    if way.refs.len() < 2 {
                        return Err(OsmError::Xml {
                            message: format!("way {} has fewer than 2 nodes", way.id),
                            offset: parser.offset(),
                        });
                    }
                    map.ways.push(way);
                }
            }
            XmlEvent::Text(_) => {}
        }
    }

    if let Some(way) = cur_way {
        return Err(OsmError::Xml {
            message: format!("unexpected end of input inside way {}", way.id),
            offset: parser.offset(),
        });
    }
    if skip_depth > 0 {
        return Err(OsmError::Xml {
            message: "unexpected end of input inside skipped subtree".into(),
            offset: parser.offset(),
        });
    }
    Ok(map)
}

/// Parse an OSM XML fragment (byte slice; input must be UTF-8).
pub fn parse_osm_bytes(bytes: &[u8]) -> Result<OsmMap, OsmError> {
    let text = std::str::from_utf8(bytes).map_err(|e| OsmError::Xml {
        message: format!("input is not valid UTF-8: {e}"),
        offset: e.valid_up_to(),
    })?;
    parse_osm(text)
}

fn req_attr<'v>(
    attrs: &'v [(std::borrow::Cow<'_, str>, std::borrow::Cow<'_, str>)],
    element: &'static str,
    attr: &'static str,
) -> Result<&'v str, OsmError> {
    attrs
        .iter()
        .find(|(k, _)| &**k == attr)
        .map(|(_, v)| v.as_ref())
        .ok_or(OsmError::MissingAttr { element, attr })
}

fn req_attr_i64(
    attrs: &[(std::borrow::Cow<'_, str>, std::borrow::Cow<'_, str>)],
    element: &'static str,
    attr: &'static str,
) -> Result<i64, OsmError> {
    let raw = req_attr(attrs, element, attr)?;
    raw.parse::<i64>()
        .map_err(|_| OsmError::Number {
            value: raw.to_string(),
            what: match attr {
                "id" => "id",
                "ref" => "node ref",
                _ => "integer attribute",
            },
        })
}

fn req_attr_f64(
    attrs: &[(std::borrow::Cow<'_, str>, std::borrow::Cow<'_, str>)],
    element: &'static str,
    attr: &'static str,
) -> Result<f64, OsmError> {
    let raw = req_attr(attrs, element, attr)?;
    raw.parse::<f64>()
        .map_err(|_| OsmError::Number {
            value: raw.to_string(),
            what: match attr {
                "lat" => "latitude",
                "lon" => "longitude",
                _ => "coordinate",
            },
        })
}

fn parse_bounds(
    attrs: &[(std::borrow::Cow<'_, str>, std::borrow::Cow<'_, str>)],
) -> Result<OsmBounds, OsmError> {
    let vals = ["minlat", "minlon", "maxlat", "maxlon"];
    let mut got = [None::<f64>; 4];
    for (i, name) in vals.iter().enumerate() {
        if let Some((_, raw)) = attrs.iter().find(|(k, _)| &**k == *name) {
            match raw.parse::<f64>() {
                Ok(v) => got[i] = Some(v),
                Err(_) => {
                    return Err(OsmError::Number {
                        value: raw.to_string(),
                        what: "bounds coordinate",
                    })
                }
            }
        } else {
            return Err(OsmError::MissingAttr {
                element: "bounds",
                attr: match *name {
                    "minlat" => "minlat",
                    "minlon" => "minlon",
                    "maxlat" => "maxlat",
                    _ => "maxlon",
                },
            });
        }
    }
    Ok(OsmBounds {
        min_lat: got[0].expect("checked above"),
        min_lon: got[1].expect("checked above"),
        max_lat: got[2].expect("checked above"),
        max_lon: got[3].expect("checked above"),
    })
}

// ---------------------------------------------------------------------------
// Extractors
// ---------------------------------------------------------------------------

/// A road polyline (resolved way with a `highway` tag).
#[derive(Debug, Clone, PartialEq)]
pub struct Road {
    /// Source way id.
    pub id: i64,
    /// Raw `highway=*` value (`"primary"`, `"residential"`, ...).
    pub kind: String,
    /// `name=*` tag when present.
    pub name: Option<String>,
    /// Resolved `(lat, lon)` points in way order.
    pub points: Vec<(f64, f64)>,
}

/// A building footprint (closed way tagged `building`).
#[derive(Debug, Clone, PartialEq)]
pub struct Building {
    /// Source way id.
    pub id: i64,
    /// Footprint vertices, closing duplicate removed.
    pub points: Vec<(f64, f64)>,
    /// `height=*` in metres when present and parseable.
    pub height_m: Option<f64>,
    /// `name=*` tag when present.
    pub name: Option<String>,
}

impl OsmMap {
    /// All ways tagged `highway=*`, resolved to points.
    ///
    /// Ways with fewer than two resolved points are skipped (degenerate
    /// geometry); a missing node reference is a hard error.
    pub fn roads(&self) -> Result<Vec<Road>, OsmError> {
        let mut out = Vec::new();
        for way in &self.ways {
            let Some(kind) = way.tag("highway") else {
                continue;
            };
            let points = self.resolve_points(way)?;
            if points.len() < 2 {
                continue;
            }
            out.push(Road {
                id: way.id,
                kind: kind.to_string(),
                name: way.tag("name").map(str::to_string),
                points,
            });
        }
        Ok(out)
    }

    /// Closed ways tagged `building=*` (value `no` excluded), resolved to
    /// footprints with the closing vertex removed.
    ///
    /// An unclosed building way or a missing node reference is a hard
    /// error — such data would silently corrupt collision geometry.
    pub fn buildings(&self) -> Result<Vec<Building>, OsmError> {
        let mut out = Vec::new();
        for way in &self.ways {
            let Some(tag) = way.tag("building") else {
                continue;
            };
            if tag == "no" {
                continue;
            }
            let pts = self.resolve_points(way)?;
            if pts.len() < 4 || pts.first() != pts.last() {
                return Err(OsmError::Xml {
                    message: format!("building way {} is not a closed ring", way.id),
                    offset: 0,
                });
            }
            let mut points = pts;
            points.pop(); // drop the duplicated closing vertex
            out.push(Building {
                id: way.id,
                points,
                height_m: way.tag("height").and_then(parse_height),
                name: way.tag("name").map(str::to_string),
            });
        }
        Ok(out)
    }
}

/// Parse an OSM `height` tag: `"<number> [unit]"` where unit is one of
/// `m`/`meter(s)`/`metre(s)` (default) or `ft`/`foot`/`feet`/`'`
/// (converted at 0.3048 m/ft). Returns `None` for anything else.
pub fn parse_height(raw: &str) -> Option<f64> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    let split = t
        .char_indices()
        .find(|(_, c)| !(c.is_ascii_digit() || matches!(c, '.' | '+' | '-' | 'e' | 'E')))
        .map(|(i, _)| i)
        .unwrap_or(t.len());
    let (num, rest) = t.split_at(split);
    let value: f64 = num.trim().parse().ok()?;
    let unit = rest.trim().to_ascii_lowercase();
    let metres = match unit.as_str() {
        "" | "m" | "meter" | "meters" | "metre" | "metres" => value,
        "ft" | "foot" | "feet" | "'" => value * 0.3048,
        _ => return None,
    };
    (metres.is_finite() && metres >= 0.0).then_some(metres)
}

impl Building {
    /// Point-in-footprint test (ray casting, `(lat, lon)` degrees).
    pub fn contains(&self, lat: f64, lon: f64) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut inside = false;
        let mut j = n - 1;
        for i in 0..n {
            let (yi, xi) = self.points[i];
            let (yj, xj) = self.points[j];
            let crosses = (yi > lat) != (yj > lat)
                && lon < (xj - xi) * (lat - yi) / (yj - yi) + xi;
            if crosses {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="openmsfs-cleanroom">
  <bounds minlat="51.50" minlon="-0.12" maxlat="51.51" maxlon="-0.10"/>
  <node id="1" lat="51.5000" lon="-0.1100"/>
  <node id="2" lat="51.5000" lon="-0.1000"/>
  <node id="3" lat="51.5050" lon="-0.1000"/>
  <node id="4" lat="51.5050" lon="-0.1100"/>
  <node id="5" lat="51.5020" lon="-0.1080">
    <tag k="highway" v="traffic_signals"/>
  </node>
  <way id="10">
    <nd ref="1"/>
    <nd ref="2"/>
    <tag k="highway" v="primary"/>
    <tag k="name" v="A&amp;B Street"/>
  </way>
  <way id="11">
    <nd ref="1"/>
    <nd ref="2"/>
    <nd ref="3"/>
    <nd ref="4"/>
    <nd ref="1"/>
    <tag k="building" v="yes"/>
    <tag k="height" v="12.5 m"/>
    <tag k="name" v="Hangar &lt;3"/>
  </way>
  <relation id="99">
    <member type="way" ref="10" role="outer"/>
    <tag k="type" v="multipolygon"/>
  </relation>
</osm>
"#;

    #[test]
    fn fixture_parses_nodes_ways_bounds_and_skips_relations() {
        let map = parse_osm(FIXTURE).unwrap();
        let (nodes, ways) = map.counts();
        assert_eq!(nodes, 5);
        assert_eq!(ways, 2);
        assert_eq!(
            map.bounds,
            Some(OsmBounds {
                min_lat: 51.50,
                min_lon: -0.12,
                max_lat: 51.51,
                max_lon: -0.10,
            })
        );
        let road = &map.ways[0];
        assert_eq!(road.id, 10);
        assert_eq!(road.refs, vec![1, 2]);
        assert_eq!(road.tag("highway"), Some("primary"));
        assert_eq!(road.tag("name"), Some("A&B Street"));
        let node5 = map.nodes.get(&5).unwrap();
        assert_eq!((node5.lat, node5.lon), (51.5020, -0.1080));
    }

    #[test]
    fn road_extraction_resolves_points_and_names() {
        let map = parse_osm(FIXTURE).unwrap();
        let roads = map.roads().unwrap();
        assert_eq!(roads.len(), 1);
        let r = &roads[0];
        assert_eq!(r.kind, "primary");
        assert_eq!(r.name.as_deref(), Some("A&B Street"));
        assert_eq!(r.points.len(), 2);
        assert_eq!(r.points[0], (51.5000, -0.1100));
        assert_eq!(r.points[1], (51.5000, -0.1000));
    }

    #[test]
    fn building_extraction_drops_closing_vertex_and_parses_height() {
        let map = parse_osm(FIXTURE).unwrap();
        let buildings = map.buildings().unwrap();
        assert_eq!(buildings.len(), 1);
        let b = &buildings[0];
        assert_eq!(b.points.len(), 4);
        assert_eq!(b.height_m, Some(12.5));
        assert_eq!(b.name.as_deref(), Some("Hangar <3"));
        assert!(b.contains(51.5025, -0.1050));
        assert!(!b.contains(51.5090, -0.1050));
        assert!(!b.contains(51.5025, -0.1150));
    }

    #[test]
    fn self_closing_tags_comments_cdata_and_prolog_are_handled() {
        let input = "<?xml version=\"1.0\"?><!-- note --><osm>
            <node id=\"7\" lat='1.5' lon='2.5'><![CDATA[ignored]]></node>
            <!DOCTYPE osm>
            </osm>";
        let map = parse_osm(input).unwrap();
        assert_eq!(map.nodes.len(), 1);
        assert_eq!(map.nodes.get(&7).unwrap().lon, 2.5);
    }

    #[test]
    fn numeric_and_hex_entities_unescape() {
        let input = r#"<osm><way id="1"><nd ref="1"/><nd ref="2"/>
            <tag k="name" v="caf&#233; &#x1F600; &amp; co"/></way>
            <node id="1" lat="0" lon="0"/><node id="2" lat="0" lon="1"/></osm>"#;
        let map = parse_osm(input).unwrap();
        assert_eq!(
            map.ways[0].tag("name"),
            Some("café \u{1F600} & co")
        );
    }

    #[test]
    fn malformed_inputs_report_offsets_and_missing_pieces() {
        let missing_lat = r#"<osm><node id="1" lon="0"/></osm>"#;
        assert_eq!(
            parse_osm(missing_lat).unwrap_err(),
            OsmError::MissingAttr {
                element: "node",
                attr: "lat"
            }
        );

        let bad_id = r#"<osm><node id="abc" lat="0" lon="0"/></osm>"#;
        assert!(matches!(
            parse_osm(bad_id).unwrap_err(),
            OsmError::Number { value, what } if value == "abc" && what == "id"
        ));

        let bad_lat = r#"<osm><node id="1" lat="x" lon="0"/></osm>"#;
        assert!(matches!(
            parse_osm(bad_lat).unwrap_err(),
            OsmError::Number { what: "latitude", .. }
        ));

        let stray_nd = r#"<osm><nd ref="1"/></osm>"#;
        assert!(matches!(
            parse_osm(stray_nd).unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("outside")
        ));

        let eof = r#"<osm><way id="1"><nd ref="1"/>"#;
        assert!(matches!(
            parse_osm(eof).unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("unexpected end")
        ));

        let one_node_way = r#"<osm><way id="1"><nd ref="1"/></way><node id="1" lat="0" lon="0"/></osm>"#;
        assert!(matches!(
            parse_osm(one_node_way).unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("fewer than 2")
        ));

        let unterminated = r#"<osm><node id="1" lat="0" lon="0""#;
        assert!(matches!(
            parse_osm(unterminated).unwrap_err(),
            OsmError::Xml { .. }
        ));

        let bad_entity = r#"<osm><node id="1" lat="0" lon="0" note="&nope;"/></osm>"#;
        assert!(matches!(
            parse_osm(bad_entity).unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("entity")
        ));
    }

    #[test]
    fn missing_node_reference_fails_resolution() {
        let input = r#"<osm><way id="1"><nd ref="1"/><nd ref="2"/>
            <tag k="highway" v="service"/>
            <tag k="building" v="yes"/></way>
            <node id="1" lat="0" lon="0"/></osm>"#;
        let map = parse_osm(input).unwrap();
        assert_eq!(map.roads().unwrap_err(), OsmError::MissingRef(2));
        assert_eq!(map.buildings().unwrap_err(), OsmError::MissingRef(2));
    }

    #[test]
    fn building_no_is_skipped_unclosed_building_errors() {
        let no = r#"<osm><way id="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>
            <tag k="building" v="no"/></way>
            <node id="1" lat="0" lon="0"/><node id="2" lat="0" lon="1"/>
            <node id="3" lat="1" lon="1"/></osm>"#;
        let map = parse_osm(no).unwrap();
        assert!(map.buildings().unwrap().is_empty());

        let open = r#"<osm><way id="1"><nd ref="1"/><nd ref="2"/><nd ref="3"/>
            <tag k="building" v="yes"/></way>
            <node id="1" lat="0" lon="0"/><node id="2" lat="0" lon="1"/>
            <node id="3" lat="1" lon="1"/></osm>"#;
        let map = parse_osm(open).unwrap();
        assert!(matches!(
            map.buildings().unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("closed ring")
        ));
    }

    #[test]
    fn height_parser_covers_units_and_rejects_garbage() {
        assert_eq!(parse_height("12"), Some(12.0));
        assert_eq!(parse_height(" 12.5 m "), Some(12.5));
        assert_eq!(parse_height("12m"), Some(12.0));
        assert_eq!(parse_height("12 metres"), Some(12.0));
        assert_eq!(parse_height("40 ft"), Some(40.0 * 0.3048));
        assert_eq!(parse_height("40 feet"), Some(40.0 * 0.3048));
        assert_eq!(parse_height("10'"), Some(10.0 * 0.3048));
        assert_eq!(parse_height(""), None);
        assert_eq!(parse_height("tall"), None);
        assert_eq!(parse_height("12 furlongs"), None);
        assert_eq!(parse_height("-5 m"), None);
    }

    #[test]
    fn empty_input_yields_empty_map() {
        let map = parse_osm("").unwrap();
        assert_eq!(map.counts(), (0, 0));
        assert!(map.bounds.is_none());
        assert!(map.roads().unwrap().is_empty());
        assert!(map.buildings().unwrap().is_empty());
    }

    #[test]
    fn parse_osm_bytes_requires_utf8() {
        let good = parse_osm_bytes(FIXTURE.as_bytes()).unwrap();
        assert_eq!(good.counts(), (5, 2));
        let bad = parse_osm_bytes(&[0xFF, 0xFE, 0xFD]).unwrap_err();
        assert!(matches!(bad, OsmError::Xml { .. }));
    }

    #[test]
    fn duplicate_bounds_is_rejected() {
        let input = r#"<osm>
            <bounds minlat="0" minlon="0" maxlat="1" maxlon="1"/>
            <bounds minlat="0" minlon="0" maxlat="1" maxlon="1"/>
            </osm>"#;
        assert!(matches!(
            parse_osm(input).unwrap_err(),
            OsmError::Xml { message, .. } if message.contains("duplicate")
        ));
    }
}

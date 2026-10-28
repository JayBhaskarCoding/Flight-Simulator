//! OpenSimConnect-style TCP server: an external-client interface for
//! the SimVar bus.
//!
//! **Clean-room scope.** The wire format implemented here is *our own*
//! documented protocol modelled on the publicly described behaviour of
//! OpenSimConnect (connect → define a data area → request updates at a
//! period → receive responses). It is deliberately not claimed to be
//! byte-identical to any proprietary client: compatibility is defined
//! by the public interaction model, not by copied implementation.
//!
//! # Wire format
//!
//! Every message is a frame: `u32 BE payload_len`, then `payload_len`
//! bytes of `u8 type` + body. Payloads are capped at [`MAX_FRAME`].
//!
//! | dir | type | body |
//! |-----|------|------|
//! | c→s | `0x01` HELLO   | `u32 version`, `u16 name_len`, name |
//! | s→c | `0x81` WELCOME | `u32 version`, `u32 client_id` |
//! | c→s | `0x02` DEFINE  | `u16 count`, count × {`u16 name_len`, name, `u16 unit_len`, unit} |
//! | s→c | `0x82` DEFINED | `u16 count` |
//! | c→s | `0x03` REQUEST | `u32 req_id`, `u32 period_ms` (0 = one shot) |
//! | s→c | `0x83` DATA    | `u32 req_id`, `u32 seq`, `u32 period_ms`, `u16 count`, count × `f64 LE` |
//! | c→s | `0x04` STOP    | `u32 req_id` |
//! | s→c | `0x84` STOPPED | `u32 req_id` |
//! | c→s | `0x7F` BYE     | (empty) |
//! | s→c | `0x0F` ERROR   | `u16 code`, `u16 msg_len`, message |
//!
//! Units are chosen by the client per variable (`""` = raw SI of the
//! variable's dimension). Periodic requests stream DATA every
//! `period_ms` (granularity [`TICK_MS`]) until STOP or disconnect.

use std::collections::HashMap;
use std::io;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use crate::simvar::{SimVarBus, Unit};

/// Default TCP port of the SimConnect-over-IP model.
pub const DEFAULT_PORT: u16 = 45170;
/// Protocol version spoken by this server.
pub const PROTOCOL_VERSION: u32 = 1;
/// Maximum accepted payload length (type byte included).
pub const MAX_FRAME: usize = 1 << 20;
/// Streaming granularity for periodic subscriptions.
pub const TICK_MS: u64 = 10;

/// Message type bytes.
pub mod msg {
    pub const HELLO: u8 = 0x01;
    pub const DEFINE: u8 = 0x02;
    pub const REQUEST: u8 = 0x03;
    pub const STOP: u8 = 0x04;
    pub const BYE: u8 = 0x7F;
    pub const WELCOME: u8 = 0x81;
    pub const DEFINED: u8 = 0x82;
    pub const DATA: u8 = 0x83;
    pub const STOPPED: u8 = 0x84;
    pub const ERROR: u8 = 0x0F;
}

/// Error codes carried in ERROR frames.
pub mod err {
    /// A requested variable does not exist on the bus.
    pub const UNKNOWN_VAR: u16 = 1;
    /// The unit string is unknown or dimensionally wrong.
    pub const BAD_UNIT: u16 = 2;
    /// Request received in an invalid state (no define, bad id, …).
    pub const BAD_REQUEST: u16 = 3;
    /// Unsupported protocol version in HELLO.
    pub const VERSION: u16 = 4;
    /// Malformed message body.
    pub const MALFORMED: u16 = 5;
}

/// Server-side errors (framing/protocol level).
#[derive(Debug)]
pub enum OpenSimConnectError {
    Io(io::Error),
    /// Frame length exceeds [`MAX_FRAME`].
    FrameTooLarge(usize),
    /// Client sent a byte sequence that does not parse.
    Malformed(String),
}

impl std::fmt::Display for OpenSimConnectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenSimConnectError::Io(e) => write!(f, "i/o error: {e}"),
            OpenSimConnectError::FrameTooLarge(n) => {
                write!(f, "frame of {n} bytes exceeds limit {MAX_FRAME}")
            }
            OpenSimConnectError::Malformed(m) => write!(f, "malformed message: {m}"),
        }
    }
}

impl std::error::Error for OpenSimConnectError {}

impl From<io::Error> for OpenSimConnectError {
    fn from(e: io::Error) -> Self {
        OpenSimConnectError::Io(e)
    }
}

/// Map a client unit label to a [`Unit`]; `None` means "raw SI".
fn parse_unit(label: &str) -> Result<Option<Unit>, String> {
    let l = label.trim().to_ascii_lowercase();
    Ok(Some(match l.as_str() {
        "" | "si" => return Ok(None),
        "m" => Unit::Meter,
        "ft" => Unit::Foot,
        "nmi" => Unit::NauticalMile,
        "m/s" | "mps" => Unit::MetersPerSecond,
        "kt" | "knot" | "knots" => Unit::Knot,
        "mph" => Unit::MilePerHour,
        "fpm" | "ft/min" => Unit::FeetPerMinute,
        "ft/s" | "fps" => Unit::FeetPerSecond,
        "g" => Unit::GForce,
        "deg" | "°" => Unit::Degree,
        "rad" => Unit::Radian,
        "deg/s" => Unit::DegreesPerSecond,
        "rad/s" => Unit::RadiansPerSecond,
        "rpm" => Unit::RevolutionsPerMinute,
        "kg" => Unit::Kilogram,
        "lb" | "lbs" => Unit::Pound,
        "n" => Unit::Newton,
        "lbf" => Unit::PoundForce,
        "pa" => Unit::Pascal,
        "hpa" => Unit::Hectopascal,
        "inhg" => Unit::InchHg,
        "k" => Unit::Kelvin,
        "c" | "degc" | "celsius" => Unit::Celsius,
        "f" | "degf" | "fahrenheit" => Unit::Fahrenheit,
        "m3" | "m^3" => Unit::CubicMeter,
        "l" | "liter" => Unit::Liter,
        "gal" | "usgal" => Unit::GallonUS,
        "s" => Unit::Second,
        "min" => Unit::Minute,
        "h" | "hour" => Unit::Hour,
        "hz" => Unit::Hertz,
        "%" | "percent" => Unit::Percent,
        "ratio" => Unit::Ratio,
        "1" | "unitless" => Unit::Unitless,
        other => return Err(format!("unknown unit `{other}`")),
    }))
}

/// The TCP front-end for a [`SimVarBus`].
pub struct OpenSimConnectServer {
    listener: TcpListener,
    bus: SimVarBus,
}

impl OpenSimConnectServer {
    /// Bind to `addr` (e.g. `0.0.0.0:45170`).
    pub async fn bind(addr: &str, bus: SimVarBus) -> io::Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        Ok(Self { listener, bus })
    }

    /// The actual bound address (useful with port 0 in tests).
    pub fn local_addr(&self) -> io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept loop: each connection is handled independently. Runs
    /// until the listener fails (or is dropped by a task abort).
    pub async fn serve(self) -> io::Result<()> {
        loop {
            let (stream, _peer) = self.listener.accept().await?;
            let bus = self.bus.clone();
            tokio::spawn(async move {
                let _ = handle_connection(stream, bus).await;
            });
        }
    }
}

/// Per-connection protocol state.
#[derive(Default)]
struct ConnState {
    client_id: u32,
    defined: Vec<(String, Option<Unit>)>,
    subscriptions: HashMap<u32, Subscription>,
    seq: u32,
}

struct Subscription {
    period_ms: u32,
    vars: Vec<(String, Option<Unit>)>,
    next_due: tokio::time::Instant,
}

/// Cancellation-safe frame accumulator: `select!` may drop the read
/// future at any moment, so bytes are only ever consumed one `read()`
/// at a time into this buffer and frames are parsed from buffered data.
#[derive(Default)]
struct FrameBuf {
    buf: Vec<u8>,
}

impl FrameBuf {
    fn extend(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Extract the next complete frame, or `None` while incomplete.
    fn take(&mut self) -> Result<Option<Vec<u8>>, OpenSimConnectError> {
        if self.buf.len() < 4 {
            return Ok(None);
        }
        let len = u32::from_be_bytes(self.buf[0..4].try_into().unwrap()) as usize;
        if len == 0 || len > MAX_FRAME {
            return Err(OpenSimConnectError::FrameTooLarge(len));
        }
        if self.buf.len() < 4 + len {
            return Ok(None);
        }
        let frame = self.buf[4..4 + len].to_vec();
        self.buf.drain(..4 + len);
        Ok(Some(frame))
    }
}

async fn handle_connection(stream: TcpStream, bus: SimVarBus) -> io::Result<()> {
    let (mut rx, mut tx) = stream.into_split();
    let mut state = ConnState::default();
    let mut frames = FrameBuf::default();
    let mut chunk = [0u8; 4096];
    let mut tick = tokio::time::interval(Duration::from_millis(TICK_MS));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    'outer: loop {
        tokio::select! {
            // ---- client bytes ------------------------------------------
            read = rx.read(&mut chunk) => {
                match read {
                    Ok(0) => break, // orderly EOF
                    Ok(n) => frames.extend(&chunk[..n]),
                    Err(e) => return Err(e),
                }
                loop {
                    let frame = match frames.take() {
                        Ok(Some(f)) => f,
                        Ok(None) => break, // incomplete: wait for more
                        Err(e) => {
                            // framing error: report and close
                            let mut body = Vec::new();
                            body.extend_from_slice(&err::MALFORMED.to_be_bytes());
                            let m = e.to_string().into_bytes();
                            body.extend_from_slice(&(m.len() as u16).to_be_bytes());
                            body.extend_from_slice(&m);
                            let _ = send_frame(&mut tx, msg::ERROR, &body).await;
                            break 'outer;
                        }
                    };
                    if dispatch(&frame, &mut state, &bus, &mut tx).await? {
                        break 'outer; // BYE
                    }
                }
            }
            // ---- periodic streaming -------------------------------------
            _ = tick.tick() => {
                let now = tokio::time::Instant::now();
                let due: Vec<u32> = state
                    .subscriptions
                    .iter()
                    .filter(|(_, s)| s.next_due <= now)
                    .map(|(id, _)| *id)
                    .collect();
                for id in due {
                    let (period, values) = {
                        let s = state.subscriptions.get_mut(&id).expect("due id");
                        s.next_due = now + Duration::from_millis(s.period_ms.max(TICK_MS as u32) as u64);
                        (s.period_ms, values_for(&s.vars, &bus))
                    };
                    state.seq = state.seq.wrapping_add(1);
                    let body = data_body(id, state.seq, period, &values);
                    send_frame(&mut tx, msg::DATA, &body).await?;
                }
            }
        }
    }
    Ok(())
}

/// Returns `true` when the connection should close (BYE).
async fn dispatch<W: AsyncWriteExt + Unpin>(
    frame: &[u8],
    state: &mut ConnState,
    bus: &SimVarBus,
    tx: &mut W,
) -> io::Result<bool> {
    if frame.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "empty frame"));
    }
    let (kind, body) = (frame[0], &frame[1..]);
    match kind {
        msg::HELLO => {
            if body.len() < 6 {
                return send_error(tx, err::MALFORMED, "short HELLO").await.map(|_| false);
            }
            let version = u32::from_be_bytes(body[0..4].try_into().unwrap());
            if version != PROTOCOL_VERSION {
                return send_error(tx, err::VERSION, "unsupported protocol version")
                    .await
                    .map(|_| false);
            }
            state.client_id = next_client_id();
            let mut out = Vec::with_capacity(8);
            out.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
            out.extend_from_slice(&state.client_id.to_be_bytes());
            send_frame(tx, msg::WELCOME, &out).await.map(|_| false)
        }
        msg::DEFINE => {
            match parse_define(body) {
                Err((code, m)) => send_error(tx, code, &m).await.map(|_| false),
                Ok(entries) => {
                    // validate every name + unit against the bus
                    for (name, unit) in &entries {
                        if !bus.contains(name) {
                            let m = format!("unknown SimVar `{name}`");
                            return send_error(tx, err::UNKNOWN_VAR, &m).await.map(|_| false);
                        }
                        if let Some(u) = unit {
                            if let Err(e) = bus.get(name, *u) {
                                let m = e.to_string();
                                return send_error(tx, err::BAD_UNIT, &m).await.map(|_| false);
                            }
                        }
                    }
                    let count = entries.len() as u16;
                    state.defined = entries;
                    let out = count.to_be_bytes();
                    send_frame(tx, msg::DEFINED, &out[..]).await.map(|_| false)
                }
            }
        }
        msg::REQUEST => {
            if body.len() != 8 {
                return send_error(tx, err::MALFORMED, "REQUEST body must be 8 bytes")
                    .await
                    .map(|_| false);
            }
            let req_id = u32::from_be_bytes(body[0..4].try_into().unwrap());
            let period_ms = u32::from_be_bytes(body[4..8].try_into().unwrap());
            if state.defined.is_empty() {
                return send_error(tx, err::BAD_REQUEST, "DEFINE data before REQUEST")
                    .await
                    .map(|_| false);
            }
            if period_ms == 0 {
                // one-shot response, immediate
                state.seq = state.seq.wrapping_add(1);
                let values = values_for(&state.defined, bus);
                let body = data_body(req_id, state.seq, 0, &values);
                send_frame(tx, msg::DATA, &body).await?;
                return Ok(false);
            }
            if state.subscriptions.contains_key(&req_id) {
                return send_error(tx, err::BAD_REQUEST, "request id already active")
                    .await
                    .map(|_| false);
            }
            state.subscriptions.insert(
                req_id,
                Subscription {
                    period_ms,
                    vars: state.defined.clone(),
                    next_due: tokio::time::Instant::now()
                        + Duration::from_millis(period_ms.max(TICK_MS as u32) as u64),
                },
            );
            Ok(false)
        }
        msg::STOP => {
            if body.len() != 4 {
                return send_error(tx, err::MALFORMED, "STOP body must be 4 bytes")
                    .await
                    .map(|_| false);
            }
            let req_id = u32::from_be_bytes(body[0..4].try_into().unwrap());
            if state.subscriptions.remove(&req_id).is_none() {
                return send_error(tx, err::BAD_REQUEST, "unknown request id")
                    .await
                    .map(|_| false);
            }
            send_frame(tx, msg::STOPPED, &req_id.to_be_bytes()).await.map(|_| false)
        }
        msg::BYE => Ok(true),
        other => {
            let m = format!("unknown message type 0x{other:02X}");
            send_error(tx, err::MALFORMED, &m).await.map(|_| false)
        }
    }
}

fn values_for(vars: &[(String, Option<Unit>)], bus: &SimVarBus) -> Vec<f64> {
    vars.iter()
        .map(|(name, unit)| match unit {
            Some(u) => bus.get(name, *u).unwrap_or(f64::NAN),
            None => bus.get_si(name).unwrap_or(f64::NAN),
        })
        .collect()
}

fn data_body(req_id: u32, seq: u32, period_ms: u32, values: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(14 + values.len() * 8);
    b.extend_from_slice(&req_id.to_be_bytes());
    b.extend_from_slice(&seq.to_be_bytes());
    b.extend_from_slice(&period_ms.to_be_bytes());
    b.extend_from_slice(&(values.len() as u16).to_be_bytes());
    for v in values {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

/// Structural failures carry `MALFORMED`; unknown-unit failures carry
/// `BAD_UNIT` so clients can tell a bad wire format from bad input.
type DefineResult = Result<Vec<(String, Option<Unit>)>, (u16, String)>;

fn parse_define(body: &[u8]) -> DefineResult {
    if body.len() < 2 {
        return Err((err::MALFORMED, "short DEFINE".into()));
    }
    let count = u16::from_be_bytes([body[0], body[1]]);
    let mut pos = 2usize;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = take_str(body, &mut pos).map_err(|m| (err::MALFORMED, m))?;
        let unit_raw = take_str(body, &mut pos).map_err(|m| (err::MALFORMED, m))?;
        let unit = parse_unit(&unit_raw)
            .map_err(|m| (err::BAD_UNIT, format!("{name}: {m}")))?;
        out.push((name, unit));
    }
    if pos != body.len() {
        return Err((
            err::MALFORMED,
            format!("{} trailing bytes", body.len() - pos),
        ));
    }
    Ok(out)
}

fn take_str(body: &[u8], pos: &mut usize) -> Result<String, String> {
    if *pos + 2 > body.len() {
        return Err("truncated string length".into());
    }
    let len = u16::from_be_bytes([body[*pos], body[*pos + 1]]) as usize;
    *pos += 2;
    if *pos + len > body.len() {
        return Err("truncated string body".into());
    }
    let s = std::str::from_utf8(&body[*pos..*pos + len])
        .map_err(|_| "string is not utf-8".to_string())?
        .to_string();
    *pos += len;
    Ok(s)
}

async fn send_frame<W: AsyncWriteExt + Unpin>(
    tx: &mut W,
    kind: u8,
    body: &[u8],
) -> io::Result<()> {
    let payload_len = (body.len() + 1) as u32;
    tx.write_all(&payload_len.to_be_bytes()).await?;
    tx.write_all(&[kind]).await?;
    if !body.is_empty() {
        tx.write_all(body).await?;
    }
    tx.flush().await
}

async fn send_error<W: AsyncWriteExt + Unpin>(
    tx: &mut W,
    code: u16,
    message: &str,
) -> io::Result<bool> {
    let m = message.as_bytes();
    let mut body = Vec::with_capacity(4 + m.len());
    body.extend_from_slice(&code.to_be_bytes());
    body.extend_from_slice(&(m.len() as u16).to_be_bytes());
    body.extend_from_slice(m);
    send_frame(tx, msg::ERROR, &body).await?;
    Ok(false)
}

fn next_client_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static NEXT: AtomicU32 = AtomicU32::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_parser_covers_common_labels() {
        assert_eq!(parse_unit("").unwrap(), None);
        assert_eq!(parse_unit("si").unwrap(), None);
        assert_eq!(parse_unit("KT").unwrap(), Some(Unit::Knot));
        assert_eq!(parse_unit("ft").unwrap(), Some(Unit::Foot));
        assert_eq!(parse_unit("deg").unwrap(), Some(Unit::Degree));
        assert_eq!(parse_unit("fpm").unwrap(), Some(Unit::FeetPerMinute));
        assert_eq!(parse_unit("inhg").unwrap(), Some(Unit::InchHg));
        assert!(parse_unit("parsec").is_err());
    }

    #[test]
    fn define_parser_round_trip() {
        let mut body = Vec::new();
        body.extend_from_slice(&2u16.to_be_bytes());
        for (n, u) in [("INDICATED AIRSPEED", "kt"), ("ALTITUDE", "ft")] {
            body.extend_from_slice(&(n.len() as u16).to_be_bytes());
            body.extend_from_slice(n.as_bytes());
            body.extend_from_slice(&(u.len() as u16).to_be_bytes());
            body.extend_from_slice(u.as_bytes());
        }
        let parsed = parse_define(&body).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].0, "INDICATED AIRSPEED");
        assert_eq!(parsed[0].1, Some(Unit::Knot));
        assert_eq!(parsed[1].1, Some(Unit::Foot));
        // trailing garbage rejected
        let mut extra = body.clone();
        extra.push(0);
        assert!(parse_define(&extra).is_err());
    }

    #[test]
    fn data_body_layout() {
        let b = data_body(7, 3, 25, &[1.5, -2.0]);
        assert_eq!(&b[0..4], &7u32.to_be_bytes());
        assert_eq!(&b[4..8], &3u32.to_be_bytes());
        assert_eq!(&b[8..12], &25u32.to_be_bytes());
        assert_eq!(&b[12..14], &2u16.to_be_bytes());
        assert_eq!(b.len(), 14 + 16);
        let v0 = f64::from_le_bytes(b[14..22].try_into().unwrap());
        assert_eq!(v0, 1.5);
    }
}

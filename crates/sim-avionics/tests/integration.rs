//! End-to-end tests: SimVar bus ↔ autopilot ↔ gauge host ↔ TCP server.

use std::time::Duration;

use sim_avionics::gauge::GaugeHost;
use sim_avionics::opensimconnect::{err, msg, OpenSimConnectServer, PROTOCOL_VERSION};
use sim_avionics::simvar::{SimVarBus, Unit};
use sim_avionics::{ApMode, ApSample, Autopilot};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Bind a server on an ephemeral port and spawn its accept loop.
async fn start_server(bus: SimVarBus) -> (std::net::SocketAddr, tokio::task::JoinHandle<std::io::Result<()>>) {
    let server = OpenSimConnectServer::bind("127.0.0.1:0", bus).await.expect("bind");
    let addr = server.local_addr().expect("local_addr");
    let handle = tokio::spawn(server.serve());
    (addr, handle)
}

async fn send_frame(stream: &mut TcpStream, kind: u8, body: &[u8]) {
    let len = (body.len() + 1) as u32;
    stream.write_all(&len.to_be_bytes()).await.unwrap();
    stream.write_all(&[kind]).await.unwrap();
    if !body.is_empty() {
        stream.write_all(body).await.unwrap();
    }
    stream.flush().await.unwrap();
}

async fn recv_frame(stream: &mut TcpStream) -> std::io::Result<(u8, Vec<u8>)> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;
    assert!(len > 0 && len <= sim_avionics::opensimconnect::MAX_FRAME, "len={len}");
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok((buf[0], buf[1..].to_vec()))
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

async fn hello(stream: &mut TcpStream) -> u32 {
    let mut body = Vec::new();
    body.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    put_str(&mut body, "itest-client");
    send_frame(stream, msg::HELLO, &body).await;
    let (kind, resp) = recv_frame(stream).await.unwrap();
    assert_eq!(kind, msg::WELCOME);
    u32::from_be_bytes(resp[4..8].try_into().unwrap())
}

async fn define(stream: &mut TcpStream, vars: &[(&str, &str)]) {
    let mut body = Vec::new();
    body.extend_from_slice(&(vars.len() as u16).to_be_bytes());
    for (n, u) in vars {
        put_str(&mut body, n);
        put_str(&mut body, u);
    }
    send_frame(stream, msg::DEFINE, &body).await;
}

fn parse_data(body: &[u8]) -> (u32, u32, u32, Vec<f64>) {
    let req = u32::from_be_bytes(body[0..4].try_into().unwrap());
    let seq = u32::from_be_bytes(body[4..8].try_into().unwrap());
    let period = u32::from_be_bytes(body[8..12].try_into().unwrap());
    let count = u16::from_be_bytes(body[12..14].try_into().unwrap()) as usize;
    assert_eq!(body.len(), 14 + count * 8);
    let values = (0..count)
        .map(|i| f64::from_le_bytes(body[14 + i * 8..22 + i * 8].try_into().unwrap()))
        .collect();
    (req, seq, period, values)
}

fn error_code(body: &[u8]) -> u16 {
    u16::from_be_bytes(body[0..2].try_into().unwrap())
}

#[tokio::test]
async fn full_session_handshake_define_one_shot_request() {
    let bus = SimVarBus::with_defaults();
    bus.set("INDICATED AIRSPEED", 120.0, Unit::Knot).unwrap();
    bus.set("ALTITUDE", 3300.0, Unit::Foot).unwrap();
    let (addr, handle) = start_server(bus).await;

    let mut s = TcpStream::connect(addr).await.unwrap();
    let client_id = hello(&mut s).await;
    assert_ne!(client_id, 0);

    define(&mut s, &[("INDICATED AIRSPEED", "kt"), ("ALTITUDE", "ft")]).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    if kind == msg::ERROR {
        let code = u16::from_be_bytes(body[0..2].try_into().unwrap());
        let mlen = u16::from_be_bytes(body[2..4].try_into().unwrap()) as usize;
        panic!("unexpected ERROR code={code} msg={}", String::from_utf8_lossy(&body[4..4 + mlen]));
    }
    assert_eq!(kind, msg::DEFINED, "body={body:?}");
    assert_eq!(u16::from_be_bytes(body[0..2].try_into().unwrap()), 2);

    // one-shot request
    let mut req = Vec::new();
    req.extend_from_slice(&7u32.to_be_bytes());
    req.extend_from_slice(&0u32.to_be_bytes());
    send_frame(&mut s, msg::REQUEST, &req).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::DATA);
    let (r, seq, period, values) = parse_data(&body);
    assert_eq!(r, 7);
    assert!(seq >= 1);
    assert_eq!(period, 0);
    assert_eq!(values.len(), 2);
    // server answered in the UNITS REQUESTED BY THE CLIENT
    assert!((values[0] - 120.0).abs() < 1e-9, "{}", values[0]);
    assert!((values[1] - 3300.0).abs() < 1e-9, "{}", values[1]);

    // same request answered in raw SI
    define(&mut s, &[("INDICATED AIRSPEED", ""), ("ALTITUDE", "")]).await;
    let (kind, _) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::DEFINED);
    let mut req = Vec::new();
    req.extend_from_slice(&8u32.to_be_bytes());
    req.extend_from_slice(&0u32.to_be_bytes());
    send_frame(&mut s, msg::REQUEST, &req).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::DATA);
    let (_, _, _, values) = parse_data(&body);
    assert!((values[0] - 120.0 * 1852.0 / 3600.0).abs() < 1e-9, "{}", values[0]);
    assert!((values[1] - 3300.0 * 0.3048).abs() < 1e-9, "{}", values[1]);

    // graceful close
    send_frame(&mut s, msg::BYE, &[]).await;
    let mut eof = [0u8; 1];
    let n = s.read(&mut eof).await.unwrap();
    assert_eq!(n, 0, "server should close after BYE");

    handle.abort();
}

#[tokio::test]
async fn unknown_variable_and_bad_unit_return_errors() {
    let bus = SimVarBus::with_defaults();
    let (addr, handle) = start_server(bus).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    hello(&mut s).await;

    define(&mut s, &[("DEFINITELY NOT A VAR", "kt")]).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    assert_eq!(error_code(&body), err::UNKNOWN_VAR);

    // right name, dimensionally impossible unit
    define(&mut s, &[("ALTITUDE", "kt")]).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    if error_code(&body) == err::MALFORMED {
        let mlen = u16::from_be_bytes(body[2..4].try_into().unwrap()) as usize;
        panic!("framing error: {}", String::from_utf8_lossy(&body[4..4 + mlen]));
    }
    assert_eq!(error_code(&body), err::BAD_UNIT);

    // nonsense unit string
    define(&mut s, &[("ALTITUDE", "parsec")]).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    assert_eq!(error_code(&body), err::BAD_UNIT);

    // the connection survives errors: a good define still works
    define(&mut s, &[("ALTITUDE", "ft")]).await;
    let (kind, _) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::DEFINED);

    // REQUEST before DEFINE on a fresh connection
    let mut s2 = TcpStream::connect(addr).await.unwrap();
    hello(&mut s2).await;
    let mut req = Vec::new();
    req.extend_from_slice(&1u32.to_be_bytes());
    req.extend_from_slice(&0u32.to_be_bytes());
    send_frame(&mut s2, msg::REQUEST, &req).await;
    let (kind, body) = recv_frame(&mut s2).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    assert_eq!(error_code(&body), err::BAD_REQUEST);

    handle.abort();
}

#[tokio::test]
async fn wrong_protocol_version_is_rejected() {
    let (addr, handle) = start_server(SimVarBus::new()).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    let mut body = Vec::new();
    body.extend_from_slice(&99u32.to_be_bytes());
    put_str(&mut body, "old-client");
    send_frame(&mut s, msg::HELLO, &body).await;
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    assert_eq!(error_code(&body), err::VERSION);
    handle.abort();
}

#[tokio::test]
async fn periodic_subscription_streams_updates_then_stops() {
    let bus = SimVarBus::with_defaults();
    let (addr, handle) = start_server(bus.clone()).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    hello(&mut s).await;
    define(&mut s, &[("INDICATED AIRSPEED", "kt")]).await;
    let _ = recv_frame(&mut s).await.unwrap();

    // subscribe at 30 ms
    let mut req = Vec::new();
    req.extend_from_slice(&42u32.to_be_bytes());
    req.extend_from_slice(&30u32.to_be_bytes());
    send_frame(&mut s, msg::REQUEST, &req).await;

    // values should be observed changing as we write to the bus
    bus.set("INDICATED AIRSPEED", 100.0, Unit::Knot).unwrap();
    let mut seqs = Vec::new();
    let mut last_value = -1.0;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
    while seqs.len() < 3 && tokio::time::Instant::now() < deadline {
        let (kind, body) = tokio::time::timeout(Duration::from_secs(1), recv_frame(&mut s))
            .await
            .expect("timed out waiting for DATA")
            .unwrap();
        assert_eq!(kind, msg::DATA);
        let (r, seq, period, values) = parse_data(&body);
        assert_eq!(r, 42);
        assert_eq!(period, 30);
        seqs.push(seq);
        last_value = values[0];
    }
    assert!(seqs.len() >= 3, "expected >=3 frames, got {}", seqs.len());
    assert!(seqs.windows(2).all(|w| w[1] > w[0]), "seqs must increase");
    assert!((last_value - 100.0).abs() < 1e-9, "{last_value}");

    // Stop the subscription. A DATA frame already in flight may beat
    // STOPPED onto the wire, so wait for STOPPED among them.
    send_frame(&mut s, msg::STOP, &42u32.to_be_bytes()).await;
    loop {
        let (kind, body) = tokio::time::timeout(Duration::from_secs(1), recv_frame(&mut s))
            .await
            .expect("STOPPED should arrive promptly")
            .unwrap();
        if kind == msg::STOPPED {
            assert_eq!(u32::from_be_bytes(body[0..4].try_into().unwrap()), 42);
            break;
        }
        assert_eq!(kind, msg::DATA, "unexpected frame kind while stopping");
    }
    while let Ok(Ok((kind, body))) =
        tokio::time::timeout(Duration::from_millis(250), recv_frame(&mut s)).await
    {
        if kind == msg::DATA {
            let (r, _, _, _) = parse_data(&body);
            assert_ne!(r, 42, "DATA after STOP");
        }
    }

    handle.abort();
}

#[tokio::test]
async fn two_clients_get_distinct_ids_and_isolated_views() {
    let bus = SimVarBus::with_defaults();
    let (addr, handle) = start_server(bus).await;

    let mut a = TcpStream::connect(addr).await.unwrap();
    let mut b = TcpStream::connect(addr).await.unwrap();
    let id_a = hello(&mut a).await;
    let id_b = hello(&mut b).await;
    assert_ne!(id_a, id_b);

    // client A defines a var B never asked for; requests stay per-client
    define(&mut a, &[("GROSS WEIGHT", "lb")]).await;
    let (kind, _) = recv_frame(&mut a).await.unwrap();
    assert_eq!(kind, msg::DEFINED);
    let mut req = Vec::new();
    req.extend_from_slice(&1u32.to_be_bytes());
    req.extend_from_slice(&0u32.to_be_bytes());
    send_frame(&mut a, msg::REQUEST, &req).await;
    let (kind, body) = recv_frame(&mut a).await.unwrap();
    assert_eq!(kind, msg::DATA);
    let (r, _, _, values) = parse_data(&body);
    assert_eq!(r, 1);
    assert_eq!(values.len(), 1);
    assert_eq!(values[0], 0.0); // default catalog value

    handle.abort();
}

#[tokio::test]
async fn malformed_frames_close_the_connection_with_an_error() {
    let (addr, handle) = start_server(SimVarBus::new()).await;
    let mut s = TcpStream::connect(addr).await.unwrap();
    // frame claiming 2 MiB (> MAX_FRAME)
    s.write_all(&(sim_avionics::opensimconnect::MAX_FRAME as u32 + 1).to_be_bytes())
        .await
        .unwrap();
    // server replies ERROR then closes
    let (kind, body) = recv_frame(&mut s).await.unwrap();
    assert_eq!(kind, msg::ERROR);
    assert_eq!(error_code(&body), err::MALFORMED);
    let mut eof = [0u8; 1];
    let n = s.read(&mut eof).await.unwrap();
    assert_eq!(n, 0);
    handle.abort();
}

#[tokio::test]
async fn autopilot_steers_from_bus_snapshot() {
    let bus = SimVarBus::with_defaults();
    let mut ap = Autopilot::with_defaults();

    bus.set("HEADING DEGREES", 0.0, Unit::Degree).unwrap();
    bus.set("PITCH DEGREES", 0.0, Unit::Degree).unwrap();
    bus.set("ROLL DEGREES", 0.0, Unit::Degree).unwrap();
    bus.set("ALTITUDE", 5000.0, Unit::Foot).unwrap();
    bus.set("VERTICAL SPEED", 0.0, Unit::FeetPerMinute).unwrap();

    let read = |bus: &SimVarBus| ApSample {
        heading_deg: bus.get("HEADING DEGREES", Unit::Degree).unwrap(),
        pitch_deg: bus.get("PITCH DEGREES", Unit::Degree).unwrap(),
        roll_deg: bus.get("ROLL DEGREES", Unit::Degree).unwrap(),
        altitude_ft: bus.get("ALTITUDE", Unit::Foot).unwrap(),
        vertical_speed_fpm: bus.get("VERTICAL SPEED", Unit::FeetPerMinute).unwrap(),
    };

    let sample = read(&bus);
    ap.set_mode(ApMode::HeadingHold, &sample);
    ap.set_heading_target(90.0);
    let out = ap.step(&sample, 0.02);
    assert!(out.aileron > 0.0, "right turn expected, got {}", out.aileron);
    assert!(out.roll_reference_deg > 0.0);

    // bank right => ROLL DEGREES becomes positive; loop then reduces
    // the aileron command as the roll closes on the reference.
    bus.set("ROLL DEGREES", out.roll_reference_deg, Unit::Degree).unwrap();
    let sample2 = read(&bus);
    let out2 = ap.step(&sample2, 0.02);
    // roll reached the reference: P-term vanishes and the derivative
    // term actively opposes further rotation (negative aileron)
    assert!(
        out2.aileron < 0.0,
        "on-reference roll must be damped, got aileron {} (was {})",
        out2.aileron,
        out.aileron
    );
}

#[test]
fn gauge_consumes_bus_values_end_to_end() {
    // A smoothing gauge fed from the bus: the gauge is the *display*
    // side of the same data the autopilot consumes.
    let wat = r#"
        (module
          (global $in (mut f64) (f64.const 0))
          (global $filt (mut f64) (f64.const 0))
          (func (export "gauge_inputs") (result i32) (i32.const 1))
          (func (export "gauge_outputs") (result i32) (i32.const 1))
          (func (export "set_input") (param $i i32) (param $v f64)
            (global.set $in (local.get $v)))
          (func (export "get_output") (param $i i32) (result f64) (global.get $filt))
          (func (export "tick") (param $dt f64)
            (global.set $filt
              (f64.add (global.get $filt)
                (f64.mul (f64.const 0.5)
                  (f64.sub (global.get $in) (global.get $filt)))))))
    "#;
    let host = GaugeHost::new(wat.as_bytes()).unwrap();
    let mut gauge = host.instantiate().unwrap();
    let bus = SimVarBus::with_defaults();
    bus.set("INDICATED AIRSPEED", 100.0, Unit::Knot).unwrap();

    let ias = bus.get("INDICATED AIRSPEED", Unit::MetersPerSecond).unwrap();
    let out = gauge.run(&[ias], 0.02).unwrap();
    assert!((out[0] - ias * 0.5).abs() < 1e-9);
    let out = gauge.run(&[ias], 0.02).unwrap();
    assert!((out[0] - ias * 0.75).abs() < 1e-9);
}

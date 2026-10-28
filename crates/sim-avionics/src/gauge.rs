//! Wasmtime gauge host: runs instrument gauge modules.
//!
//! A gauge is a WebAssembly module exporting exactly this ABI (all
//! signatures are validated when the host loads the module):
//!
//! | export          | signature      | meaning                          |
//! |-----------------|----------------|----------------------------------|
//! | `gauge_inputs`  | `() -> i32`    | number of input channels         |
//! | `gauge_outputs` | `() -> i32`    | number of output channels        |
//! | `set_input`     | `(i32, f64)`   | write input `i` (bounds-checked) |
//! | `get_output`    | `(i32) -> f64` | read output `i`                  |
//! | `tick`          | `(f64)`        | advance by `dt` seconds          |
//!
//! The host owns the engine and module; each [`GaugeInstance`] carries
//! its own `Store` (gauge state lives in module globals or memory), so
//! several instruments can run the same compiled module independently.

use wasmtime::{Engine, Func, Instance, Linker, Module, Store, TypedFunc};

/// Errors raised by the gauge host.
#[derive(Debug, PartialEq, Eq)]
pub enum GaugeError {
    /// The bytes are not a valid WebAssembly module (or not valid wat).
    Module(String),
    /// Instantiation failed (link error, trap during start, …).
    Instantiate(String),
    /// A required ABI export is absent.
    MissingExport(String),
    /// An export exists with the wrong signature.
    AbiMismatch(String),
    /// `run` was called with the wrong number of inputs.
    InputCount { expected: usize, got: usize },
    /// The module reported a nonsense channel count.
    BadChannelCount(String),
}

impl std::fmt::Display for GaugeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GaugeError::Module(e) => write!(f, "invalid gauge module: {e}"),
            GaugeError::Instantiate(e) => write!(f, "gauge instantiation failed: {e}"),
            GaugeError::MissingExport(n) => write!(f, "gauge is missing export `{n}`"),
            GaugeError::AbiMismatch(n) => write!(f, "gauge export `{n}` has the wrong signature"),
            GaugeError::InputCount { expected, got } => {
                write!(f, "gauge expects {expected} inputs, got {got}")
            }
            GaugeError::BadChannelCount(e) => write!(f, "gauge channel count invalid: {e}"),
        }
    }
}

impl std::error::Error for GaugeError {}

const MAX_CHANNELS: i32 = 4096;

/// Compiled gauge module shared across instances.
pub struct GaugeHost {
    engine: Engine,
    module: Module,
    inputs: usize,
    outputs: usize,
}

impl std::fmt::Debug for GaugeHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GaugeHost")
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish_non_exhaustive()
    }
}

impl GaugeHost {
    /// Compile and ABI-validate a gauge from raw bytes (wat text is
    /// accepted because the `wat` feature is enabled).
    pub fn new(wasm: &[u8]) -> Result<Self, GaugeError> {
        let engine = Engine::default();
        let module = Module::new(&engine, wasm)
            .map_err(|e| GaugeError::Module(e.to_string()))?;
        // A throwaway instance validates every required export.
        let mut probe = Store::new(&engine, ());
        let linker = Linker::new(&engine);
        let inst = linker
            .instantiate(&mut probe, &module)
            .map_err(|e| GaugeError::Instantiate(e.to_string()))?;
        let inputs_fn = typed_export::< (), i32>(&mut probe, &inst, "gauge_inputs")?;
        let outputs_fn = typed_export::<(), i32>(&mut probe, &inst, "gauge_outputs")?;
        typed_export::<(i32, f64), ()>(&mut probe, &inst, "set_input")?;
        typed_export::<i32, f64>(&mut probe, &inst, "get_output")?;
        typed_export::<f64, ()>(&mut probe, &inst, "tick")?;
        let inputs = inputs_fn
            .call(&mut probe, ())
            .map_err(|e| GaugeError::BadChannelCount(e.to_string()))?;
        let outputs = outputs_fn
            .call(&mut probe, ())
            .map_err(|e| GaugeError::BadChannelCount(e.to_string()))?;
        if !(0..=MAX_CHANNELS).contains(&inputs) || !(0..=MAX_CHANNELS).contains(&outputs) {
            return Err(GaugeError::BadChannelCount(format!(
                "inputs={inputs} outputs={outputs} (max {MAX_CHANNELS})"
            )));
        }
        Ok(Self {
            engine,
            module,
            inputs: inputs as usize,
            outputs: outputs as usize,
        })
    }

    /// Input channels declared by the module.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Output channels declared by the module.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Create an independent instance (fresh module state).
    pub fn instantiate(&self) -> Result<GaugeInstance, GaugeError> {
        let mut store = Store::new(&self.engine, ());
        let linker = Linker::new(&self.engine);
        let instance = linker
            .instantiate(&mut store, &self.module)
            .map_err(|e| GaugeError::Instantiate(e.to_string()))?;
        let inputs_fn = typed_export::<(i32, f64), ()>(&mut store, &instance, "set_input")?;
        let outputs_fn = typed_export::<i32, f64>(&mut store, &instance, "get_output")?;
        let tick_fn = typed_export::<f64, ()>(&mut store, &instance, "tick")?;
        Ok(GaugeInstance {
            store,
            instance,
            inputs_fn,
            outputs_fn,
            tick_fn,
            inputs: self.inputs,
            outputs: self.outputs,
        })
    }
}

/// Fetch an export and verify its exact wasm signature.
fn typed_export<P, R>(
    store: &mut Store<()>,
    instance: &Instance,
    name: &str,
) -> Result<TypedFunc<P, R>, GaugeError>
where
    P: wasmtime::WasmParams,
    R: wasmtime::WasmResults,
{
    // Missing vs mistyped are different diagnostics for the user.
    let generic: Func = instance
        .get_func(&mut *store, name)
        .ok_or_else(|| GaugeError::MissingExport(name.to_string()))?;
    let _ = generic; // existence checked; typed fetch validates signature
    instance
        .get_typed_func::<P, R>(store, name)
        .map_err(|_| GaugeError::AbiMismatch(name.to_string()))
}

/// One running gauge with its own state.
pub struct GaugeInstance {
    store: Store<()>,
    instance: Instance,
    inputs_fn: TypedFunc<(i32, f64), ()>,
    outputs_fn: TypedFunc<i32, f64>,
    tick_fn: TypedFunc<f64, ()>,
    inputs: usize,
    outputs: usize,
}

impl GaugeInstance {
    /// Number of input channels (mirrors the host declaration).
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Number of output channels (mirrors the host declaration).
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Feed `inputs` (must match the declared count), advance the gauge
    /// by `dt` seconds and read back all output channels.
    pub fn run(&mut self, inputs: &[f64], dt: f64) -> Result<Vec<f64>, GaugeError> {
        if inputs.len() != self.inputs {
            return Err(GaugeError::InputCount {
                expected: self.inputs,
                got: inputs.len(),
            });
        }
        for (i, v) in inputs.iter().enumerate() {
            self.inputs_fn
                .call(&mut self.store, (i as i32, *v))
                .map_err(|e| GaugeError::BadChannelCount(e.to_string()))?;
        }
        self.tick_fn
            .call(&mut self.store, dt)
            .map_err(|e| GaugeError::BadChannelCount(e.to_string()))?;
        let mut out = Vec::with_capacity(self.outputs);
        for i in 0..self.outputs {
            let v = self
                .outputs_fn
                .call(&mut self.store, i as i32)
                .map_err(|e| GaugeError::BadChannelCount(e.to_string()))?;
            out.push(v);
        }
        Ok(out)
    }

    /// Raw access to the underlying store for advanced embedding.
    pub fn store(&mut self) -> &mut Store<()> {
        &mut self.store
    }

    /// The instantiated module.
    pub fn instance(&self) -> Instance {
        self.instance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smoothing gauge: one input, one output, EMA with alpha = 0.3.
    const EMA_GAUGE: &str = r#"
        (module
          (global $in (mut f64) (f64.const 0))
          (global $filt (mut f64) (f64.const 0))
          (func (export "gauge_inputs") (result i32) (i32.const 1))
          (func (export "gauge_outputs") (result i32) (i32.const 1))
          (func (export "set_input") (param $i i32) (param $v f64)
            (if (i32.eqz (local.get $i))
              (then (global.set $in (local.get $v)))))
          (func (export "get_output") (param $i i32) (result f64)
            (global.get $filt))
          (func (export "tick") (param $dt f64)
            (global.set $filt
              (f64.add (global.get $filt)
                (f64.mul (f64.const 0.3)
                  (f64.sub (global.get $in) (global.get $filt)))))))
    "#;

    #[test]
    fn ema_gauge_smooths_towards_input() {
        let host = GaugeHost::new(EMA_GAUGE.as_bytes()).unwrap();
        assert_eq!(host.inputs(), 1);
        assert_eq!(host.outputs(), 1);
        let mut g = host.instantiate().unwrap();
        let o1 = g.run(&[100.0], 0.02).unwrap();
        assert!((o1[0] - 30.0).abs() < 1e-9, "{}", o1[0]);
        let o2 = g.run(&[100.0], 0.02).unwrap();
        assert!((o2[0] - 51.0).abs() < 1e-9, "{}", o2[0]);
        for _ in 0..60 {
            g.run(&[100.0], 0.02).unwrap();
        }
        let last = g.run(&[100.0], 0.02).unwrap();
        assert!((last[0] - 100.0).abs() < 1e-3, "{}", last[0]);
    }

    #[test]
    fn instances_do_not_share_state() {
        let host = GaugeHost::new(EMA_GAUGE.as_bytes()).unwrap();
        let mut a = host.instantiate().unwrap();
        let mut b = host.instantiate().unwrap();
        for _ in 0..10 {
            a.run(&[1000.0], 0.02).unwrap();
        }
        let b_out = b.run(&[1000.0], 0.02).unwrap();
        assert!((b_out[0] - 300.0).abs() < 1e-9, "fresh instance saw {}", b_out[0]);
        // `a` kept its own state: EMA after 11 ticks at alpha = 0.3 is
        // 1000 * (1 - 0.7^11) = 980.2267…
        let a_out = a.run(&[1000.0], 0.02).unwrap();
        let expected = 1000.0 * (1.0 - 0.7_f64.powi(11));
        assert!((a_out[0] - expected).abs() < 1e-6, "a={}", a_out[0]);
    }

    #[test]
    fn input_count_mismatch_is_rejected() {
        let host = GaugeHost::new(EMA_GAUGE.as_bytes()).unwrap();
        let mut g = host.instantiate().unwrap();
        let err = g.run(&[1.0, 2.0], 0.01).unwrap_err();
        assert_eq!(
            err,
            GaugeError::InputCount {
                expected: 1,
                got: 2
            }
        );
    }

    #[test]
    fn missing_export_is_reported() {
        let wat = r#"
            (module
              (func (export "gauge_inputs") (result i32) (i32.const 1))
              (func (export "gauge_outputs") (result i32) (i32.const 1))
              (func (export "set_input") (param i32 f64))
              (func (export "get_output") (param i32) (result f64) (f64.const 0)))
        "#;
        let err = GaugeHost::new(wat.as_bytes()).unwrap_err();
        assert_eq!(err, GaugeError::MissingExport("tick".into()));
    }

    #[test]
    fn wrong_signature_is_reported() {
        let wat = r#"
            (module
              (func (export "gauge_inputs") (result i32) (i32.const 1))
              (func (export "gauge_outputs") (result i32) (i32.const 1))
              (func (export "set_input") (param i32 f64))
              (func (export "get_output") (param i32) (result f64) (f64.const 0))
              (func (export "tick") (param i32)))
        "#;
        let err = GaugeHost::new(wat.as_bytes()).unwrap_err();
        assert_eq!(err, GaugeError::AbiMismatch("tick".into()));
    }

    #[test]
    fn invalid_bytes_are_reported() {
        let err = GaugeHost::new(b"\0asm\x01\0\0\0garbage").unwrap_err();
        assert!(matches!(err, GaugeError::Module(_)), "{err:?}");
        let err2 = GaugeHost::new(b"not wat at all").unwrap_err();
        assert!(matches!(err2, GaugeError::Module(_)), "{err2:?}");
    }

    #[test]
    fn multi_channel_gauge() {
        // 2 inputs (hi, lo selection), 1 output: picks the larger.
        let wat = r#"
            (module
              (global $a (mut f64) (f64.const 0))
              (global $b (mut f64) (f64.const 0))
              (global $out (mut f64) (f64.const 0))
              (func (export "gauge_inputs") (result i32) (i32.const 2))
              (func (export "gauge_outputs") (result i32) (i32.const 1))
              (func (export "set_input") (param $i i32) (param $v f64)
                (if (i32.eqz (local.get $i))
                  (then (global.set $a (local.get $v)))
                  (else (global.set $b (local.get $v)))))
              (func (export "get_output") (param $i i32) (result f64)
                (global.get $out))
              (func (export "tick") (param $dt f64)
                (global.set $out
                  (if (result f64) (f64.gt (global.get $a) (global.get $b))
                    (then (global.get $a))
                    (else (global.get $b))))))
        "#;
        let host = GaugeHost::new(wat.as_bytes()).unwrap();
        let mut g = host.instantiate().unwrap();
        assert_eq!(g.run(&[3.0, 7.0], 0.01).unwrap(), vec![7.0]);
        assert_eq!(g.run(&[9.0, 7.0], 0.01).unwrap(), vec![9.0]);
    }
}

//! Embedded WGSL sources + offline validation.
//!
//! Every shader is validated at test time through `naga`'s WGSL
//! frontend **and** its module validator, so a syntax or
//! type/address-space error fails `cargo test` even without a GPU.

/// Physically-based forward pass (vertex + fragment).
pub const PBR_WGSL: &str = include_str!("shaders/pbr.wgsl");
/// Hillaire sky background pass (fullscreen).
pub const SKY_WGSL: &str = include_str!("shaders/sky.wgsl");
/// Volumetric cloud raymarch pass (fullscreen).
pub const CLOUDS_WGSL: &str = include_str!("shaders/clouds.wgsl");
/// ACES tonemap / display pass (fullscreen).
pub const TONEMAP_WGSL: &str = include_str!("shaders/tonemap.wgsl");

/// (label, source) pairs for [`validate_all`].
pub fn all_shaders() -> [(&'static str, &'static str); 4] {
    [
        ("pbr", PBR_WGSL),
        ("sky", SKY_WGSL),
        ("clouds", CLOUDS_WGSL),
        ("tonemap", TONEMAP_WGSL),
    ]
}

/// Parse + validate one WGSL module with naga (no GPU required).
pub fn validate(label: &str, source: &str) -> Result<(), String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("{label}: parse error: {}", e.emit_to_string(source)))?;
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    );
    validator
        .validate(&module)
        .map_err(|e| format!("{label}: validation error: {e:?}"))?;
    Ok(())
}

/// Validate every embedded shader.
pub fn validate_all() -> Result<(), String> {
    for (label, src) in all_shaders() {
        validate(label, src)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_wgsl_module_parses_and_validates() {
        if let Err(e) = validate_all() {
            panic!("{e}");
        }
    }

    #[test]
    fn shader_sources_are_non_empty() {
        for (label, src) in all_shaders() {
            assert!(src.len() > 400, "{label} suspiciously small");
        }
    }

    #[test]
    fn malformed_wgsl_is_rejected() {
        assert!(validate("broken", "@vertex fn vs() -> @builtin(position) f32 {").is_err());
    }
}

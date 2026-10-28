//! Parsed cfg values: numbers, numeric vectors, and free text.

use crate::error::{CfgError, CfgResult};

/// A parsed cfg value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A single number (integers and floats alike).
    Number(f64),
    /// A comma-separated list of numbers (`1.0, 2.0, 3.0`).
    Vector(Vec<f64>),
    /// Free text: quoted strings, boolean words, or anything else.
    Text(String),
}

impl Value {
    /// Parse a raw value fragment (already comment-stripped).
    pub fn parse(raw: &str) -> Value {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Value::Text(String::new());
        }
        // quoted string
        if (trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2)
            || (trimmed.starts_with('\'') && trimmed.ends_with('\'') && trimmed.len() >= 2)
        {
            return Value::Text(trimmed[1..trimmed.len() - 1].to_string());
        }
        // numeric vector?
        if trimmed.contains(',') {
            let mut nums = Vec::new();
            let mut all_numbers = true;
            for part in trimmed.split(',') {
                let p = part.trim();
                if p.is_empty() {
                    all_numbers = false;
                    break;
                }
                match p.parse::<f64>() {
                    Ok(v) => nums.push(v),
                    Err(_) => {
                        all_numbers = false;
                        break;
                    }
                }
            }
            if all_numbers {
                return Value::Vector(nums);
            }
            return Value::Text(trimmed.to_string());
        }
        // single number?
        if let Ok(v) = trimmed.parse::<f64>() {
            return Value::Number(v);
        }
        Value::Text(trimmed.to_string())
    }

    /// The value as a scalar, or a type error naming `key`/`line`.
    pub fn as_f64(&self, key: &str, line: usize, file: &str) -> CfgResult<f64> {
        match self {
            Value::Number(v) => Ok(*v),
            Value::Vector(v) if v.len() == 1 => Ok(v[0]),
            other => Err(CfgError::Type {
                file: file.to_string(),
                line,
                key: key.to_string(),
                expected: "a number",
                found: other.render(),
            }),
        }
    }

    /// The value as a numeric vector.
    pub fn as_vec(&self, key: &str, line: usize, file: &str) -> CfgResult<Vec<f64>> {
        match self {
            Value::Number(v) => Ok(vec![*v]),
            Value::Vector(v) => Ok(v.clone()),
            other => Err(CfgError::Type {
                file: file.to_string(),
                line,
                key: key.to_string(),
                expected: "a comma-separated numeric vector",
                found: other.render(),
            }),
        }
    }

    /// The value as text (quotes already removed by [`Value::parse`]).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Boolean interpretation: `1/0`, `true/false`, `yes/no`, `on/off`
    /// (case-insensitive). Numbers other than 0 are true.
    pub fn as_bool(&self, key: &str, line: usize, file: &str) -> CfgResult<bool> {
        match self {
            Value::Number(v) => Ok(*v != 0.0),
            Value::Vector(v) if v.len() == 1 => Ok(v[0] != 0.0),
            Value::Text(t) => match t.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => Ok(true),
                "0" | "false" | "no" | "off" => Ok(false),
                _ => Err(CfgError::Type {
                    file: file.to_string(),
                    line,
                    key: key.to_string(),
                    expected: "a boolean (0/1/true/false/yes/no/on/off)",
                    found: t.clone(),
                }),
            },
            other => Err(CfgError::Type {
                file: file.to_string(),
                line,
                key: key.to_string(),
                expected: "a boolean (0/1/true/false/yes/no/on/off)",
                found: other.render(),
            }),
        }
    }

    /// Human-readable rendering for error messages.
    pub fn render(&self) -> String {
        match self {
            Value::Number(v) => v.to_string(),
            Value::Vector(v) => v
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            Value::Text(s) => s.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_numbers() {
        assert_eq!(Value::parse("1.5"), Value::Number(1.5));
        assert_eq!(Value::parse("  -42 "), Value::Number(-42.0));
        assert_eq!(Value::parse("1e3"), Value::Number(1000.0));
        assert_eq!(Value::parse(".25"), Value::Number(0.25));
    }

    #[test]
    fn parses_vectors() {
        assert_eq!(
            Value::parse("1, 2.5, -3"),
            Value::Vector(vec![1.0, 2.5, -3.0])
        );
        // mixed content falls back to text
        assert_eq!(Value::parse("1, abc"), Value::Text("1, abc".into()));
    }

    #[test]
    fn parses_quoted_and_text() {
        assert_eq!(Value::parse("\"Cessna 172S\""), Value::Text("Cessna 172S".into()));
        assert_eq!(Value::parse("TRUE"), Value::Text("TRUE".into()));
        assert!(Value::parse("TRUE").as_bool("k", 1, "f").unwrap());
        assert!(!Value::parse("off").as_bool("k", 1, "f").unwrap());
        assert!(Value::parse("3").as_bool("k", 1, "f").unwrap());
    }

    #[test]
    fn type_errors_carry_context() {
        let e = Value::parse("abc").as_f64("wing_area", 12, "fm.cfg").unwrap_err();
        let s = e.to_string();
        assert!(s.contains("wing_area") && s.contains("12") && s.contains("fm.cfg"), "{s}");
    }
}

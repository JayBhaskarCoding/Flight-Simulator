//! Error type for cfg parsing and typed document loading.

/// All errors produced by this crate.
#[derive(Debug)]
pub enum CfgError {
    /// Filesystem failure while reading a cfg file.
    Io {
        path: String,
        source: std::io::Error,
    },
    /// Structural syntax error (malformed section header or key line).
    Syntax {
        file: String,
        line: usize,
        message: String,
    },
    /// A value could not be interpreted as the requested type.
    Type {
        file: String,
        line: usize,
        key: String,
        expected: &'static str,
        found: String,
    },
    /// A section required by a typed loader is absent.
    MissingSection {
        file: String,
        section: String,
    },
    /// A key required by a typed loader is absent or empty.
    MissingKey {
        file: String,
        section: String,
        key: String,
    },
    /// A value is syntactically valid but outside its valid range.
    Range {
        file: String,
        line: usize,
        key: String,
        message: String,
    },
    /// An indexed family (e.g. `aileron_area[0..n]`) has gaps.
    IndexGap {
        file: String,
        section: String,
        key: String,
        expected: String,
    },
}

impl CfgError {
    pub(crate) fn syntax(file: &str, line: usize, message: impl Into<String>) -> Self {
        Self::Syntax { file: file.to_string(), line, message: message.into() }
    }
}

impl std::fmt::Display for CfgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {path}: {source}"),
            Self::Syntax { file, line, message } => {
                write!(f, "{file}:{line}: syntax error: {message}")
            }
            Self::Type { file, line, key, expected, found } => write!(
                f,
                "{file}:{line}: key '{key}' expects {expected}, found '{found}'"
            ),
            Self::MissingSection { file, section } => {
                write!(f, "{file}: missing required section [{section}]")
            }
            Self::MissingKey { file, section, key } => write!(
                f,
                "{file}: section [{section}] is missing required key '{key}'"
            ),
            Self::Range { file, line, key, message } => {
                write!(f, "{file}:{line}: key '{key}': {message}")
            }
            Self::IndexGap { file, section, key, expected } => write!(
                f,
                "{file}: section [{section}]: indexed key '{key}' should have {expected}"
            ),
        }
    }
}

impl std::error::Error for CfgError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Result alias for this crate.
pub type CfgResult<T> = Result<T, CfgError>;

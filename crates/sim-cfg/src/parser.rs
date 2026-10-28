//! Structural parser for the MSFS-style cfg dialect.

use crate::error::{CfgError, CfgResult};
use crate::value::Value;
use std::collections::BTreeMap;

/// One `key = value` line.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Key as written (case preserved), without the `[index]` suffix.
    pub key: String,
    /// Optional `[n]` subscript.
    pub index: Option<usize>,
    /// Parsed value.
    pub value: Value,
    /// 1-based source line.
    pub line: usize,
}

/// A `[SECTION]` (or `[SECTION.variant]`) and its entries in file order.
#[derive(Debug, Clone)]
pub struct Section {
    /// Section name as written, e.g. `AIRPLANE_GEOMETRY` or `FUEL.left`.
    pub name: String,
    /// Entries in source order.
    pub entries: Vec<Entry>,
    /// Line of the section header.
    pub line: usize,
    /// Source file name (for error reporting).
    pub file: String,
}

impl Section {
    /// Case-insensitive lookup of an entry, ignoring the index.
    pub fn find(&self, key: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.key.eq_ignore_ascii_case(key))
    }

    /// Case-insensitive lookup of an indexed entry.
    pub fn find_indexed(&self, key: &str, index: usize) -> Option<&Entry> {
        self.entries
            .iter()
            .find(|e| e.key.eq_ignore_ascii_case(key) && e.index == Some(index))
    }

    /// All entries whose key matches, ordered by index (entries without
    /// an index sort first as index 0 candidates are rejected by callers
    /// that need explicit subscripts).
    pub fn find_all(&self, key: &str) -> Vec<&Entry> {
        self.entries
            .iter()
            .filter(|e| e.key.eq_ignore_ascii_case(key))
            .collect()
    }

    /// Scalar number (required — error if absent or non-numeric).
    pub fn f64(&self, key: &str) -> CfgResult<f64> {
        let e = self.find(key).ok_or_else(|| CfgError::MissingKey {
            file: self.file.clone(),
            section: self.name.clone(),
            key: key.to_string(),
        })?;
        e.value.as_f64(&e.key, e.line, &self.file)
    }

    /// Scalar number with a default when the key is absent; still a type
    /// error when present but malformed.
    pub fn f64_or(&self, key: &str, default: f64) -> CfgResult<f64> {
        match self.find(key) {
            None => Ok(default),
            Some(e) => e.value.as_f64(&e.key, e.line, &self.file),
        }
    }

    /// Required unsigned integer.
    pub fn u64(&self, key: &str) -> CfgResult<u64> {
        let v = self.f64(key)?;
        if v < 0.0 || v.fract() != 0.0 {
            let e = self.find(key).expect("checked above");
            return Err(CfgError::Type {
                file: self.file.clone(),
                line: e.line,
                key: e.key.clone(),
                expected: "a non-negative integer",
                found: e.value.render(),
            });
        }
        Ok(v as u64)
    }

    /// Required boolean.
    pub fn bool(&self, key: &str) -> CfgResult<bool> {
        let e = self.find(key).ok_or_else(|| CfgError::MissingKey {
            file: self.file.clone(),
            section: self.name.clone(),
            key: key.to_string(),
        })?;
        e.value.as_bool(&e.key, e.line, &self.file)
    }

    /// Boolean with default when absent.
    pub fn bool_or(&self, key: &str, default: bool) -> CfgResult<bool> {
        match self.find(key) {
            None => Ok(default),
            Some(e) => e.value.as_bool(&e.key, e.line, &self.file),
        }
    }

    /// Required free-text value.
    pub fn text(&self, key: &str) -> CfgResult<String> {
        let e = self.find(key).ok_or_else(|| CfgError::MissingKey {
            file: self.file.clone(),
            section: self.name.clone(),
            key: key.to_string(),
        })?;
        match &e.value {
            Value::Text(s) => Ok(s.clone()),
            Value::Number(v) => Ok(v.to_string()),
            Value::Vector(v) => Ok(v
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")),
        }
    }

    /// Free text with default when absent.
    pub fn text_or(&self, key: &str, default: &str) -> CfgResult<String> {
        match self.find(key) {
            None => Ok(default.to_string()),
            Some(_) => self.text(key),
        }
    }

    /// Numeric vector (required).
    pub fn vec(&self, key: &str) -> CfgResult<Vec<f64>> {
        let e = self.find(key).ok_or_else(|| CfgError::MissingKey {
            file: self.file.clone(),
            section: self.name.clone(),
            key: key.to_string(),
        })?;
        e.value.as_vec(&e.key, e.line, &self.file)
    }

    /// Required numeric vector of exactly `n` components.
    pub fn vec_n(&self, key: &str, n: usize) -> CfgResult<Vec<f64>> {
        let v = self.vec(key)?;
        if v.len() != n {
            let e = self.find(key).expect("checked above");
            return Err(CfgError::Type {
                file: self.file.clone(),
                line: e.line,
                key: e.key.clone(),
                expected: match n {
                    2 => "a 2-component vector",
                    3 => "a 3-component vector",
                    4 => "a 4-component vector",
                    _ => "a fixed-length numeric vector",
                },
                found: e.value.render(),
            });
        }
        Ok(v)
    }

    /// Collect an indexed family: returns `index -> value` for all
    /// `key[i]` entries, verifying there are no gaps below the maximum.
    pub fn indexed_f64s(&self, key: &str) -> CfgResult<BTreeMap<usize, f64>> {
        let mut out = BTreeMap::new();
        let mut max: Option<usize> = None;
        for e in self.find_all(key) {
            if let Some(i) = e.index {
                let v = e.value.as_f64(&e.key, e.line, &self.file)?;
                out.insert(i, v);
                max = Some(max.map_or(i, |m: usize| m.max(i)));
            }
        }
        if let Some(m) = max {
            for i in 0..=m {
                if !out.contains_key(&i) {
                    return Err(CfgError::IndexGap {
                        file: self.file.clone(),
                        section: self.name.clone(),
                        key: key.to_string(),
                        expected: format!("indices 0..={m} without gaps (missing [{i}])"),
                    });
                }
            }
        }
        Ok(out)
    }

    /// `indexed_f64s` but requiring at least one entry.
    pub fn indexed_f64s_required(&self, key: &str) -> CfgResult<BTreeMap<usize, f64>> {
        let m = self.indexed_f64s(key)?;
        if m.is_empty() {
            return Err(CfgError::MissingKey {
                file: self.file.clone(),
                section: self.name.clone(),
                key: format!("{key}[0]"),
            });
        }
        Ok(m)
    }

    /// Range-check helper: value must lie within `[lo, hi]`.
    pub fn f64_in(&self, key: &str, lo: f64, hi: f64) -> CfgResult<f64> {
        let v = self.f64(key)?;
        if !(lo..=hi).contains(&v) {
            let e = self.find(key).expect("checked above");
            return Err(CfgError::Range {
                file: self.file.clone(),
                line: e.line,
                key: e.key.clone(),
                message: format!("value {v} outside [{lo}, {hi}]"),
            });
        }
        Ok(v)
    }
}

/// A whole cfg document.
#[derive(Debug, Clone)]
pub struct CfgFile {
    /// Source file name (for error reporting).
    pub file: String,
    /// Sections in source order (same-name sections are merged, with
    /// entries appended — MSFS files occasionally split a section).
    pub sections: Vec<Section>,
}

impl CfgFile {
    /// Parse a cfg document.
    pub fn parse(text: &str, file: &str) -> CfgResult<Self> {
        let mut sections: Vec<Section> = Vec::new();
        // implicit root for key lines before any [SECTION] header
        let mut root = Section {
            name: String::new(),
            entries: Vec::new(),
            line: 1,
            file: file.to_string(),
        };
        // index of the section receiving entries (None = root)
        let mut current: Option<usize> = None;

        for (idx, raw_line) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = strip_comment(raw_line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') {
                let end = line.find(']').ok_or_else(|| {
                    CfgError::syntax(file, line_no, "unterminated section header")
                })?;
                let name = line[1..end].trim();
                if name.is_empty() {
                    return Err(CfgError::syntax(file, line_no, "empty section header"));
                }
                let trailing = line[end + 1..].trim();
                if !trailing.is_empty() {
                    return Err(CfgError::syntax(
                        file,
                        line_no,
                        format!("unexpected text after section header: '{trailing}'"),
                    ));
                }
                match sections.iter().position(|s| s.name.eq_ignore_ascii_case(name)) {
                    Some(i) => current = Some(i),
                    None => {
                        sections.push(Section {
                            name: name.to_string(),
                            entries: Vec::new(),
                            line: line_no,
                            file: file.to_string(),
                        });
                        current = Some(sections.len() - 1);
                    }
                }
                continue;
            }
            // key = value
            let eq = line.find('=').ok_or_else(|| {
                CfgError::syntax(
                    file,
                    line_no,
                    format!("expected 'key = value', found '{line}'"),
                )
            })?;
            let key_part = line[..eq].trim();
            let value_part = line[eq + 1..].trim();
            if key_part.is_empty() {
                return Err(CfgError::syntax(file, line_no, "empty key before '='"));
            }
            let (key, index) = split_index(key_part, file, line_no)?;
            let entry = Entry {
                key,
                index,
                value: Value::parse(value_part),
                line: line_no,
            };
            match current {
                Some(i) => sections[i].entries.push(entry),
                None => root.entries.push(entry),
            }
        }

        if !root.entries.is_empty() {
            sections.insert(0, root);
        }
        Ok(Self { file: file.to_string(), sections })
    }

    /// Case-insensitive section lookup by name.
    pub fn section(&self, name: &str) -> Option<&Section> {
        self.sections.iter().find(|s| s.name.eq_ignore_ascii_case(name))
    }

    /// Required section lookup.
    pub fn require_section(&self, name: &str) -> CfgResult<&Section> {
        self.section(name).ok_or_else(|| CfgError::MissingSection {
            file: self.file.clone(),
            section: name.to_string(),
        })
    }

    /// First section whose name starts with `prefix` (case-insensitive),
    /// used for variant sections like `FUEL.left_tank`.
    pub fn section_prefix(&self, prefix: &str) -> Option<&Section> {
        self.sections.iter().find(|s| {
            s.name
                .eq_ignore_ascii_case(prefix)
                || s.name
                    .to_ascii_lowercase()
                    .starts_with(&format!("{}.", prefix.to_ascii_lowercase()))
        })
    }
}

/// Strip a trailing/leading comment, honouring quoted strings.
fn strip_comment(line: &str) -> &str {
    let bytes = line.as_bytes();
    let mut in_quote: Option<u8> = None;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        match in_quote {
            Some(q) => {
                if b == q {
                    in_quote = None;
                }
            }
            None => {
                if b == b'"' || b == b'\'' {
                    in_quote = Some(b);
                } else if b == b'#'
                    || b == b';'
                    || (b == b'/' && bytes.get(i + 1) == Some(&b'/'))
                {
                    // `#`, `;` or `//` comment starts here
                    return &line[..i];
                }
            }
        }
        i += 1;
    }
    line
}

/// Split `key[3]` into (`key`, Some(3)).
fn split_index(key: &str, file: &str, line: usize) -> CfgResult<(String, Option<usize>)> {
    if let Some(open) = key.rfind('[') {
        if key.ends_with(']') {
            let inner = &key[open + 1..key.len() - 1];
            let name = key[..open].trim();
            if name.is_empty() {
                return Err(CfgError::syntax(file, line, "missing key before '[index]'"));
            }
            match inner.trim().parse::<usize>() {
                Ok(i) => return Ok((name.to_string(), Some(i))),
                Err(_) => {
                    return Err(CfgError::syntax(
                        file,
                        line,
                        format!("'[{inner}]' is not a valid non-negative integer index"),
                    ))
                }
            }
        }
    }
    Ok((key.to_string(), None))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
// leading comment
; alternative comment style
[SECTION]
key = 1.5 // trailing comment
other[2] = 3, 4, 5
name = "quoted ; string"
flag = TRUE
# hash comment

[SECTION.sub]
nested = -7
"#;

    #[test]
    fn parses_structure() {
        let f = CfgFile::parse(SAMPLE, "sample.cfg").unwrap();
        assert_eq!(f.sections.len(), 2);
        let s = f.require_section("section").unwrap();
        assert_eq!(s.name, "SECTION");
        assert_eq!(s.f64("key").unwrap(), 1.5);
        assert_eq!(s.vec("other").unwrap(), vec![3.0, 4.0, 5.0]);
        assert_eq!(s.find_indexed("other", 2).unwrap().value, Value::Vector(vec![3.0, 4.0, 5.0]));
        assert_eq!(s.text("name").unwrap(), "quoted ; string");
        assert!(s.bool("flag").unwrap());
        let sub = f.section("SECTION.sub").unwrap();
        assert_eq!(sub.f64("nested").unwrap(), -7.0);
    }

    #[test]
    fn case_insensitive_lookup() {
        let f = CfgFile::parse(SAMPLE, "s.cfg").unwrap();
        let s = f.section("SeCtIoN").unwrap();
        assert_eq!(s.f64("KeY").unwrap(), 1.5);
    }

    #[test]
    fn errors_carry_line_numbers() {
        let e = CfgFile::parse("[A]\nfoo bar\n", "x.cfg").unwrap_err();
        assert!(e.to_string().contains("x.cfg:2"), "{e}");

        let f = CfgFile::parse("[A]\nk = [oops]\n", "x.cfg").unwrap();
        let e = f.section("A").unwrap().f64("k").unwrap_err();
        assert!(e.to_string().contains("x.cfg:2"), "{e}");

        let e = CfgFile::parse("[A]\nk = abc\n", "x.cfg").unwrap();
        let e = e.section("A").unwrap().f64("k").unwrap_err();
        assert!(e.to_string().contains("x.cfg:2") && e.to_string().contains("expects a number"), "{e}");
    }

    #[test]
    fn missing_key_and_section_errors() {
        let f = CfgFile::parse("[A]\nk = 1\n", "x.cfg").unwrap();
        let e = f.require_section("B").unwrap_err();
        assert!(e.to_string().contains("[B]"), "{e}");
        let e = f.section("A").unwrap().f64("nope").unwrap_err();
        assert!(e.to_string().contains("'nope'"), "{e}");
    }

    #[test]
    fn indexed_families_detect_gaps() {
        let f = CfgFile::parse("[A]\nx[0] = 1\nx[2] = 3\n", "x.cfg").unwrap();
        let e = f.section("A").unwrap().indexed_f64s("x").unwrap_err();
        assert!(e.to_string().contains("missing [1]"), "{e}");

        let f = CfgFile::parse("[A]\nx[0] = 1\nx[1] = 2\n", "x.cfg").unwrap();
        let m = f.section("A").unwrap().indexed_f64s("x").unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[&1], 2.0);
    }

    #[test]
    fn merged_duplicate_sections() {
        let f = CfgFile::parse("[A]\na = 1\n[A]\nb = 2\n", "x.cfg").unwrap();
        assert_eq!(f.sections.len(), 1);
        let s = f.section("A").unwrap();
        assert_eq!(s.f64("a").unwrap(), 1.0);
        assert_eq!(s.f64("b").unwrap(), 2.0);
    }

    #[test]
    fn quoted_strings_keep_comment_markers() {
        let f = CfgFile::parse("[A]\ns = \"has // inside\"\n", "x.cfg").unwrap();
        assert_eq!(f.section("A").unwrap().text("s").unwrap(), "has // inside");
    }

    #[test]
    fn range_checks() {
        let f = CfgFile::parse("[A]\nalpha = 95\n", "x.cfg").unwrap();
        let e = f.section("A").unwrap().f64_in("alpha", -20.0, 20.0).unwrap_err();
        assert!(e.to_string().contains("outside [-20, 20]"), "{e}");
    }

    #[test]
    fn unsigned_integer_validation() {
        let f = CfgFile::parse("[A]\nn = 4\nneg = -1\nfrac = 2.5\n", "x.cfg").unwrap();
        let s = f.section("A").unwrap();
        assert_eq!(s.u64("n").unwrap(), 4);
        assert!(s.u64("neg").is_err());
        assert!(s.u64("frac").is_err());
    }
}

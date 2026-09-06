//! Every number quoted in prose is traceable to a measurement.
//!
//! `AGENTS.md` requires it and nothing enforced it, which is how `README.md`
//! came to claim 0.27 s for a re-index the measurements put at 0.29 s. This
//! reads the prose of the guides, pulls out every number carrying a unit, and
//! asks whether that number appears anywhere under `docs/measurements/`.
//!
//! **It checks existence, not agreement.** A number used in the wrong sense
//! still passes; what it catches is a value that changed in one place and not
//! the other, because the old value stops appearing. That is the failure that
//! actually happens.
//!
//! Fenced blocks are skipped. They hold command transcripts and flag values —
//! `--max-chars 8000`, a port, an example's output — which are illustrations
//! rather than claims about how folio performs. Checking them produced five
//! false alarms for every real finding when this was first tried.

use std::{fs, path::Path};

/// Units that mark a number as a claim rather than a label. Longest first, so
/// `sections` is never read as `s`.
const UNITS: &[&str] = &[
    "characters", "sections", "seconds", "tokens", "files", "rows", "ms", "MB", "KB", "GB", "%",
    "s",
];

/// The documents whose prose has to be traceable.
const GUIDES: &[&str] = &["README.md", "SKILL.md", "AGENTS.md"];

/// Drop fenced blocks, keeping the line count so a failure can name a line.
fn prose(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut fenced = false;
    for (i, line) in text.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if !fenced {
            out.push((i + 1, line.to_string()));
        }
    }
    out
}

/// Thousands separators are a formatting choice, not a different number.
fn normalize(s: &str) -> String {
    s.replace(',', "")
}

/// Every `<number> <unit>` in one line, as the number's own text.
fn claims(line: &str) -> Vec<String> {
    let bytes: Vec<char> = line.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        // A number preceded by a word character or a dot is part of something
        // else: a version, a path, an identifier.
        if i > 0 && (bytes[i - 1].is_alphanumeric() || bytes[i - 1] == '.' || bytes[i - 1] == '-') {
            while i < bytes.len() && (bytes[i].is_alphanumeric() || bytes[i] == '.') {
                i += 1;
            }
            continue;
        }
        let start = i;
        while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == ',' || bytes[i] == '.') {
            i += 1;
        }
        let number: String = bytes[start..i].iter().collect();
        let number = number.trim_end_matches(['.', ',']).to_string();
        let rest: String = bytes[i..].iter().collect();
        let rest = rest.trim_start();
        if let Some(unit) = UNITS.iter().find(|u| {
            rest.starts_with(**u)
                && rest[u.len()..]
                    .chars()
                    .next()
                    .is_none_or(|c| !c.is_alphanumeric())
        }) {
            out.push(format!("{number} {unit}"));
        }
    }
    out
}

#[test]
fn every_number_in_prose_is_traceable_to_a_measurement() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("docs/measurements");
    let Ok(entries) = fs::read_dir(&dir) else {
        // A published crate carries no docs directory. Nothing to check, and
        // nothing to fail for someone who only wanted to build folio.
        return;
    };
    let mut measured = String::new();
    for entry in entries.flatten() {
        if entry.path().extension().is_some_and(|e| e == "md") {
            measured.push_str(&fs::read_to_string(entry.path()).unwrap());
        }
    }
    assert!(
        !measured.is_empty(),
        "{} holds no measurements, so nothing could be traced to it",
        dir.display()
    );
    let measured = normalize(&measured);

    let mut untraceable = Vec::new();
    for guide in GUIDES {
        let Ok(text) = fs::read_to_string(root.join(guide)) else {
            continue;
        };
        for (line, content) in prose(&text) {
            for claim in claims(&content) {
                let number = claim.split(' ').next().unwrap();
                if !measured.contains(&normalize(number)) {
                    untraceable.push(format!("{guide}:{line}  {claim}"));
                }
            }
        }
    }
    assert!(
        untraceable.is_empty(),
        "these numbers are quoted in prose and appear in no file under docs/measurements/.\n\
         Measure them and record one, or stop quoting them:\n  {}",
        untraceable.join("\n  ")
    );
}

#[test]
fn the_scanner_reads_a_claim_and_ignores_a_label() {
    assert_eq!(claims("a query over 119,565 sections is 135 ms"),
               vec!["119,565 sections", "135 ms"]);
    assert_eq!(claims("no section was cut"), Vec::<String>::new());
    assert!(claims("folio 0.4.0 and jj 0.44.0 are versions").is_empty());
    assert_eq!(claims("`--max-chars 8000` accepts 8000 characters"), vec!["8000 characters"],
               "a flag's own value is not a claim; the sentence's is");
}

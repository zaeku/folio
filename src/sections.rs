//! Splitting a markdown file into heading sections, and flattening its
//! frontmatter into filterable keys.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// One indexed unit. Holds a reference to a range of a file, never its body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    pub path: String,
    /// 1-indexed inclusive line range in the source file.
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heading: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub breadcrumb: Vec<String>,
    /// Frontmatter, flattened to dotted keys. Any producer key is kept as-is.
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub fm: Map<String, Value>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub truncated: bool,
    /// Section text, used only to build the vector.
    #[serde(skip)]
    pub text: String,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Split `source` into sections. The frontmatter block is excluded from every
/// range and attached to each section as `fm`.
pub fn split(path: &str, source: &str) -> Vec<Section> {
    let lines: Vec<&str> = source.lines().collect();
    let (fm, body_start) = frontmatter(&lines);

    let mut heads: Vec<(usize, usize, String)> = Vec::new();
    let mut fenced = false;
    for (i, line) in lines.iter().enumerate().skip(body_start) {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        if let Some((level, title)) = atx(line) {
            heads.push((i, level, title));
        }
    }

    let mut out = Vec::new();
    let first = heads.first().map_or(lines.len(), |h| h.0);
    if lines[body_start..first].iter().any(|l| !l.trim().is_empty()) {
        out.push(mk(path, body_start, first, None, &[], &fm, &lines));
    }

    let mut stack: Vec<(usize, String)> = Vec::new();
    for (k, (i, level, title)) in heads.iter().enumerate() {
        while stack.last().is_some_and(|(l, _)| *l >= *level) {
            stack.pop();
        }
        let breadcrumb: Vec<String> = stack.iter().map(|(_, t)| t.clone()).collect();
        let end = heads.get(k + 1).map_or(lines.len(), |h| h.0);
        out.push(mk(path, *i, end, Some(title.clone()), &breadcrumb, &fm, &lines));
        stack.push((*level, title.clone()));
    }
    out
}

/// Break a section into consecutive pieces that each fit the budget.
///
/// A piece keeps the heading, the trail and the frontmatter of the section it
/// came from, and carries its own line range, so a result still names something
/// a reader can open. The tail of a long section is a vector rather than
/// nothing: measured on `mdn/content`, giving a cut tail its own record took
/// retrieval of a sentence from it from 3 of 12 at mean rank 6.0 to 6 of 12 at
/// mean rank 1.5.
///
/// The boundary falls at a blank line when one fits, at a line otherwise. The
/// same corpus is why the fallback exists: its long sections are tables and
/// lists with no blank line to break at, and a paragraph rule alone reached a
/// fifth of them. A line cut mid-sentence still has a vector; a tail has none.
///
/// One line longer than the budget cannot be split by either rule. It is cut
/// and marked, which is the case D-01M1PP6HJWFT2Q still governs.
pub fn to_budget(section: Section, budget: usize) -> Vec<Section> {
    if section.text.chars().count() <= budget {
        return vec![section];
    }
    let lines: Vec<&str> = section.text.split('\n').collect();
    let cost = |i: usize| lines[i].chars().count() + 1;

    let mut out: Vec<Section> = Vec::new();
    let mut from = 0usize;
    while from < lines.len() {
        // How far the budget reaches, by whole lines.
        let mut upto = from;
        let mut size = cost(from);
        while upto + 1 < lines.len() && size + cost(upto + 1) <= budget {
            upto += 1;
            size += cost(upto);
        }
        // Prefer to end at a paragraph break, but only one near the end: an
        // earlier break would leave the rest to a piece that has to fit too,
        // and shortening a piece can never overflow it.
        if upto + 1 < lines.len() {
            let floor = from + (upto - from) * 4 / 5;
            if let Some(b) = (floor..=upto).rev().find(|&i| lines[i].trim().is_empty()) {
                upto = b;
            }
        }
        // One line longer than the budget has no boundary inside it. It is cut
        // and marked, which is the case D-01M1PP6HJWFT2Q still governs.
        let text: String = lines[from..=upto].join("\n");
        let cut = text.chars().count() > budget;
        out.push(Section {
            start: section.start + from,
            end: section.start + upto,
            text: if cut { text.chars().take(budget).collect() } else { text },
            truncated: cut,
            path: section.path.clone(),
            heading: section.heading.clone(),
            breadcrumb: section.breadcrumb.clone(),
            fm: section.fm.clone(),
        });
        from = upto + 1;
    }
    out
}

fn mk(
    path: &str,
    from: usize,
    to: usize,
    heading: Option<String>,
    breadcrumb: &[String],
    fm: &Map<String, Value>,
    lines: &[&str],
) -> Section {
    Section {
        path: path.to_string(),
        start: from + 1,
        end: to,
        heading,
        breadcrumb: breadcrumb.to_vec(),
        fm: fm.clone(),
        truncated: false,
        text: lines[from..to].join("\n"),
    }
}

/// An ATX heading: one to six `#` followed by a space. Deliberately strict, so
/// a `#!` shebang or a `#tag` does not open a section.
fn atx(line: &str) -> Option<(usize, String)> {
    let level = line.bytes().take_while(|b| *b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = line.get(level..)?;
    if !rest.starts_with(' ') {
        return None;
    }
    Some((level, rest.trim().to_string()))
}

/// Returns the flattened frontmatter and the index of the first body line.
fn frontmatter(lines: &[&str]) -> (Map<String, Value>, usize) {
    if lines.first().map(|l| l.trim_end()) != Some("---") {
        return (Map::new(), 0);
    }
    let Some(close) = (1..lines.len()).find(|&i| lines[i].trim_end() == "---") else {
        return (Map::new(), 0);
    };
    let mut flat = Map::new();
    if let Ok(v) = serde_yaml_ng::from_str::<Value>(&lines[1..close].join("\n")) {
        flatten("", &v, &mut flat);
    }
    (flat, close + 1)
}

/// Nested maps become dotted keys. A list of scalars is kept whole so a filter
/// can read it as membership; a list holding maps is indexed per element.
fn flatten(prefix: &str, v: &Value, out: &mut Map<String, Value>) {
    let key = |k: &str| {
        if prefix.is_empty() { k.to_string() } else { format!("{prefix}.{k}") }
    };
    match v {
        Value::Object(m) => {
            for (k, vv) in m {
                flatten(&key(k), vv, out);
            }
        }
        Value::Array(a) if a.iter().any(|e| e.is_object() || e.is_array()) => {
            for (i, vv) in a.iter().enumerate() {
                flatten(&format!("{prefix}[{i}]"), vv, out);
            }
        }
        _ if prefix.is_empty() => {}
        _ => {
            out.insert(prefix.to_string(), v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "---
type: Decision
id: 0019-1
status: live
tags: [alpha, beta]
verified: { by: human:zaeku, at: 2026-09-01 }
---

Preamble line.

## Top

body

```sh
# not a heading
```

### Nested

tail
";

    #[test]
    fn a_long_section_becomes_pieces_that_fit() {
        let para = "a".repeat(40);
        let source = format!("## H\n\n{para}\n\n{para}\n\n{para}\n");
        let sec = split("a.md", &source).pop().unwrap();
        let pieces = to_budget(sec, 60);

        assert!(pieces.len() > 1, "a section over the budget is more than one piece");
        for p in &pieces {
            assert!(p.text.chars().count() <= 60, "every piece fits: {}", p.text.len());
            assert_eq!(p.heading.as_deref(), Some("H"), "a piece keeps its heading");
        }
        // Consecutive and covering: a reader following the ranges reads the section.
        for w in pieces.windows(2) {
            assert_eq!(w[0].end + 1, w[1].start, "pieces are consecutive");
        }
        assert_eq!(pieces[0].start, 1, "the first piece starts where the section did");
        assert!(pieces.iter().all(|p| !p.truncated), "nothing was cut, only divided");
    }

    #[test]
    fn one_line_over_the_budget_is_cut_and_marked() {
        // No boundary exists inside a single line, so the piece is truncated
        // and says so, which is what D-01M1PP6HJWFT2Q requires.
        let source = format!("## H\n\n{}\n", "b".repeat(200));
        let sec = split("a.md", &source).pop().unwrap();
        let pieces = to_budget(sec, 50);
        assert!(pieces.iter().any(|p| p.truncated), "the unsplittable line is marked");
    }

    #[test]
    fn splits_sections_and_flattens_frontmatter() {
        let s = split("d.md", DOC);

        let headings: Vec<_> = s.iter().map(|x| x.heading.as_deref()).collect();
        assert_eq!(
            headings,
            vec![None, Some("Top"), Some("Nested")],
            "a `#` inside a fence must not open a section"
        );

        assert_eq!(s[0].start, 8, "ranges start after the frontmatter block");
        assert_eq!(s[1].end + 1, s[2].start, "ranges are contiguous and inclusive");
        assert_eq!(s[2].breadcrumb, vec!["Top".to_string()]);

        let fm = &s[0].fm;
        assert_eq!(fm["status"], Value::from("live"));
        assert_eq!(fm["verified.by"], Value::from("human:zaeku"));
        assert!(
            fm["tags"].as_array().unwrap().contains(&Value::from("alpha")),
            "scalar lists stay whole so a filter can read membership"
        );
    }

    #[test]
    fn file_without_frontmatter_keeps_line_one() {
        let s = split("d.md", "# Only\n\nbody\n");
        assert_eq!(s.len(), 1);
        assert_eq!((s[0].start, s[0].end), (1, 3));
    }
}

//! Cutting a source file into functions, for a corpus that turns `code` on.
//!
//! The grammars and the query that finds a function belong to `ast-query`, so
//! the languages folio covers are the ones that crate's `functions` template
//! covers. What is folio's is the unit: a definition widened to the comment,
//! attribute and decorator lines directly above it.

use std::path::Path;

use serde_json::Map;

use crate::sections::Section;

/// The version of the cutter, recorded in an index so that spans cut by one
/// release are not taken for spans another would cut.
pub const CUTTER: &str = ast_query::VERSION;

fn extension(path: &str) -> Option<&str> {
    Path::new(path).extension().and_then(|e| e.to_str())
}

/// Whether `path` is a source file folio can cut into functions.
pub fn is_source(path: &str) -> bool {
    extension(path)
        .and_then(ast_query::language_for_extension)
        .is_some_and(|lang| ast_query::template_query(lang, "functions").is_some())
}

/// The lines that belong to the definition below them, by language.
fn attached(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &["//", "#["],
        "python" | "bash" => &["#", "@"],
        // `#` is the preprocessor in C and C++, not a comment.
        _ => &["//", "/*", "*", "@"],
    }
}

/// Every function `source` defines, as a unit spanning the definition and the
/// lines attached above it. A definition the parser recovered from an error
/// inside is left out: error recovery returns namespaces and classes as
/// functions, and those are not one.
pub fn split(path: &str, source: &str) -> Vec<Section> {
    let Some(ext) = extension(path) else { return Vec::new() };
    let Some(lang) = ast_query::language_for_extension(ext) else { return Vec::new() };
    let Ok(spans) = ast_query::template(ext, source, "functions") else { return Vec::new() };
    let lines: Vec<&str> = source.lines().collect();
    let above = attached(lang);
    spans
        .into_iter()
        .filter(|s| !s.has_error && s.end_line <= lines.len())
        .map(|s| {
            let mut start = s.start_line;
            while start > 1 && above.iter().any(|p| lines[start - 2].trim_start().starts_with(p)) {
                start -= 1;
            }
            Section {
                path: path.to_string(),
                start,
                end: s.end_line,
                heading: s.name,
                breadcrumb: Vec::new(),
                fm: Map::new(),
                truncated: false,
                text: lines[start - 1..s.end_line].join("\n"),
            }
        })
        .collect()
}

/// The document prefix a function is embedded behind, with `{file}` naming its
/// file.
pub fn prefix(template: &str, path: &str) -> String {
    let file = Path::new(path).file_name().and_then(|f| f.to_str()).unwrap_or(path);
    template.replace("{file}", file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_function_takes_the_lines_attached_above_it() {
        let src = "use std::io;\n\n/// Adds.\n#[inline]\nfn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\nfn two() -> i32 {\n    2\n}\n";
        let units = split("src/lib.rs", src);
        let spans: Vec<_> = units.iter().map(|u| (u.start, u.end, u.heading.as_deref())).collect();
        assert_eq!(spans, vec![(3, 7, Some("add")), (9, 11, Some("two"))]);
        assert!(units[0].text.starts_with("/// Adds."));
    }

    #[test]
    fn a_blank_line_ends_what_is_attached() {
        let src = "# unrelated\n\n@cache\ndef f():\n    return 1\n";
        let units = split("m.py", src);
        assert_eq!((units[0].start, units[0].end), (3, 5));
    }

    #[test]
    fn a_preprocessor_line_is_not_attached_in_c() {
        let src = "#include <stdio.h>\nint main(void) {\n    return 0;\n}\n";
        let units = split("main.c", src);
        assert_eq!((units[0].start, units[0].end), (2, 4));
    }

    #[test]
    fn only_languages_with_a_functions_template_are_source() {
        assert!(is_source("a/b.rs") && is_source("x.py") && is_source("y.tsx"));
        assert!(!is_source("notes.md") && !is_source("data.json") && !is_source("Makefile"));
    }

    #[test]
    fn the_prefix_names_the_file_not_its_path() {
        assert_eq!(prefix("title: {file} | text: ", "src/deep/lib.rs"), "title: lib.rs | text: ");
        assert_eq!(prefix("title: none | text: ", "src/lib.rs"), "title: none | text: ");
    }
}

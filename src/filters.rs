//! The `--where` grammar: what a filter may say about one record.
//!
//! A predicate reads a record's frontmatter and nothing else, no field is built
//! in, and what the grammar cannot express it refuses rather than guessing —
//! D-01M1VKX0S2S2F7. What emptied a result is named for the same reason: the
//! alternative is a caller bisecting their own filter.

use anyhow::{Result, anyhow};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

/// One thing a filter can test about one record.
#[derive(Debug)]
enum Term {
    Eq(String, String),
    Ne(String, String),
    Has(String),
    Lacks(String),
}

/// One `--where`: alternatives, any of which satisfies it.
///
/// `=` is a comparison rather than an assignment, so it binds tighter than `|`
/// the way it does anywhere else: `status=live | status=draft` is two
/// comparisons joined by an or. The two levels never need parentheses because
/// they live in different places — `|` inside one argument is the or, and the
/// boundary between arguments is the and.
#[derive(Debug)]
pub(crate) struct Pred {
    /// As typed, so a filter that keeps nothing can be named back to its author.
    text: String,
    any: Vec<Term>,
}

/// The whole grammar, for the error that names it.
const WHERE_GRAMMAR: &str = "`key`, `!key`, `key=value`, `key!=value`, joined by `|`";

/// Characters an operator would use, and folio compares text only. A key
/// carrying one is a predicate somebody meant and folio cannot express.
///
/// Only the key is read this way. A value is literal text once `|` has split
/// the alternatives, because a value is data and refusing characters in it
/// would deny somebody a value they legitimately wrote. The cost is that a
/// value cannot contain a bare `|`, which is the same cost every language pays
/// before it has quoting.
const NOT_A_KEY: &[char] = &['<', '>', '|', '!', '=', '~', '&'];

fn parse_term(raw: &str, whole: &str) -> Result<Term> {
    let reject = |why: String| -> anyhow::Error {
        anyhow!(
            "--where {whole}: {why}. folio compares text, and the whole grammar is {WHERE_GRAMMAR}"
        )
    };
    let raw = raw.trim();
    let (term, key) = if let Some((k, v)) = raw.split_once("!=") {
        (Term::Ne(k.trim().to_string(), v.trim().to_string()), k.trim())
    } else if let Some((k, v)) = raw.split_once('=') {
        (Term::Eq(k.trim().to_string(), v.trim().to_string()), k.trim())
    } else if let Some(k) = raw.strip_prefix('!') {
        (Term::Lacks(k.trim().to_string()), k.trim())
    } else {
        (Term::Has(raw.to_string()), raw)
    };
    if key.is_empty() {
        return Err(reject("names no key".to_string()));
    }
    if let Some(c) = key.chars().find(|c| NOT_A_KEY.contains(c)) {
        return Err(reject(format!(
            "reads `{key}` as the key, and `{c}` in a key is not something you meant"
        )));
    }
    Ok(term)
}

/// Parse `--where`, refusing what folio cannot express, and hint at the one
/// reading of it that is conventional and still probably not what was meant.
///
/// A predicate that is neither understood nor refused is worse than either: it
/// silently becomes a key nobody would name. `--where version>=7` looked for a
/// key called `version>` and matched a file that had one.
pub(crate) fn parse_preds(raw: &[String]) -> Result<(Vec<Pred>, Vec<String>)> {
    let mut preds = Vec::new();
    let mut hints = Vec::new();
    for whole in raw {
        let pieces: Vec<&str> = whole.split('|').collect();
        let any: Vec<Term> =
            pieces.iter().map(|piece| parse_term(piece, whole)).collect::<Result<_>>()?;
        // `tags=alpha|beta` is `tags=alpha` or the presence of a key `beta`,
        // which is what the precedence says and is rarely what someone typing
        // it wants. A hint rather than a refusal, because the presence test is
        // a real thing to ask for and a wrong guess would deny it.
        if any.len() > 1 {
            for (piece, term) in pieces.iter().zip(&any) {
                if let Term::Has(k) = term {
                    hints.push(format!(
                        "--where {whole}: `{}` beside `|` is a presence test for the key `{k}`. \
                         A value needs its key, as in `<key>={k}`",
                        piece.trim()
                    ));
                }
            }
        }
        preds.push(Pred { text: whole.clone(), any });
    }
    Ok((preds, hints))
}

fn satisfies(fm: &Map<String, Value>, term: &Term) -> bool {
    match term {
        Term::Has(k) => fm.contains_key(k),
        Term::Lacks(k) => !fm.contains_key(k),
        Term::Eq(k, v) => fm.get(k).is_some_and(|got| holds(got, v)),
        Term::Ne(k, v) => !fm.get(k).is_some_and(|got| holds(got, v)),
    }
}

pub(crate) fn keeps(fm: &Map<String, Value>, preds: &[Pred]) -> bool {
    preds.iter().all(|p| p.any.iter().any(|t| satisfies(fm, t)))
}

/// Which `--where` kept nothing at all, for a filter that emptied the result.
///
/// "no section passed the filter" says that something did not match and not
/// what. Each predicate is applied alone, so the one that emptied the result is
/// named — including the case where a whole expression was read as one value.
pub(crate) fn blames(rows: &[(usize, Map<String, Value>)], preds: &[Pred]) -> Vec<String> {
    preds
        .iter()
        .filter(|p| !rows.iter().any(|(_, fm)| p.any.iter().any(|t| satisfies(fm, t))))
        .map(|p| p.text.clone())
        .collect()
}

fn holds(got: &Value, want: &str) -> bool {
    match got {
        Value::Array(a) => a.iter().any(|e| scalar_eq(e, want)),
        other => scalar_eq(other, want),
    }
}

fn scalar_eq(v: &Value, want: &str) -> bool {
    scalar_text(v) == want
}

/// A scalar as the text a filter compares against. A YAML `0001` that arrived
/// as a number still has to match the string somebody typed on the command
/// line, so both sides go through this.
pub(crate) fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string().trim_matches('"').to_string(),
    }
}

/// Every value any section carries under one of `keys`, flattened out of lists.
/// This is the right side of the anti-join, and it is built from every row in
/// the index rather than from the rows a filter left.
pub(crate) fn pointed_at(
    rows: &[(usize, Map<String, Value>)],
    keys: &[String],
) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (_, fm) in rows {
        for key in keys {
            match fm.get(key) {
                Some(Value::Array(a)) => out.extend(a.iter().map(scalar_text)),
                Some(v) => {
                    out.insert(scalar_text(v));
                }
                None => {}
            }
        }
    }
    out.remove("");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn row(fm: Value) -> (usize, Map<String, Value>) {
        (0, fm.as_object().expect("an object").clone())
    }
    #[test]
    fn pointed_at_flattens_lists_and_tolerates_absence() {
        let rows = vec![
            row(json!({"id": "M-2", "supersedes": ["M-1", "M-0"]})),
            row(json!({"id": "M-3", "supersedes": "M-9"})),
            row(json!({"id": "M-4"})),
        ];
        let keys = vec!["supersedes".to_string()];
        let got = pointed_at(&rows, &keys);
        assert_eq!(
            got.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["M-0", "M-1", "M-9"],
            "a list contributes every element, a scalar contributes itself, a missing key nothing"
        );
        assert!(pointed_at(&rows, &["nothing_carries_this".to_string()]).is_empty());
    }
    #[test]
    fn identity_matches_across_yaml_scalar_types() {
        // `id: 0001` can arrive as a number while the pointer arrived as a
        // string. Both sides go through scalar_text so the join still closes.
        let rows = vec![row(json!({"id": 1, "supersedes": ["1"]}))];
        let pointed = pointed_at(&rows, &["supersedes".to_string()]);
        let identity = rows[0].1.get("id").expect("an id");
        assert!(pointed.contains(&scalar_text(identity)));
    }
    #[test]
    fn an_empty_pointer_value_points_at_nothing() {
        let rows = vec![row(json!({"id": "M-1", "supersedes": ""}))];
        assert!(
            pointed_at(&rows, &["supersedes".to_string()]).is_empty(),
            "an empty value must not drop every record that has no identity"
        );
    }
    fn parse(raw: &[&str]) -> (Vec<Pred>, Vec<String>) {
        parse_preds(&raw.iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap()
    }
    fn preds(raw: &[&str]) -> Vec<Pred> {
        parse(raw).0
    }
    fn one(raw: &str) -> Result<Vec<Pred>> {
        parse_preds(&[raw.to_string()]).map(|(p, _)| p)
    }
    fn fm(json: Value) -> Map<String, Value> {
        json.as_object().unwrap().clone()
    }
    #[test]
    fn presence_and_absence_are_both_reachable() {
        let annotated = fm(json!({"status": "live"}));
        let bare = fm(json!({"title": "Beta"}));
        assert!(keeps(&annotated, &preds(&["status"])));
        assert!(!keeps(&bare, &preds(&["status"])));
        assert!(keeps(&bare, &preds(&["!status"])));
        assert!(!keeps(&annotated, &preds(&["!status"])));
    }
    #[test]
    fn a_comparison_is_refused_rather_than_read_as_a_key() {
        for attempt in ["version>=7", "version<2", "a>b|c", "a&b"] {
            let err = one(attempt)
                .expect_err(&format!("`{attempt}` has to be refused, not reinterpreted"));
            assert!(
                err.to_string().contains("the whole grammar is"),
                "the refusal must name what folio does support: {err}"
            );
        }
    }
    #[test]
    fn a_predicate_naming_no_key_is_refused() {
        assert!(one("=live").is_err());
        assert!(one("!").is_err());
    }
    #[test]
    fn a_comparison_binds_tighter_than_an_or() {
        let pred = preds(&["status=live | status=draft"]);
        assert!(keeps(&fm(json!({"status": "live"})), &pred));
        assert!(keeps(&fm(json!({"status": "draft"})), &pred));
        assert!(!keeps(&fm(json!({"status": "retired"})), &pred));
    }
    #[test]
    fn separate_flags_are_anded_and_one_flag_is_ored() {
        let pred = preds(&["status=live | status=draft", "type=guide"]);
        assert!(keeps(&fm(json!({"status": "draft", "type": "guide"})), &pred));
        assert!(
            !keeps(&fm(json!({"status": "draft", "type": "note"})), &pred),
            "the second flag has to still be required"
        );
        assert!(!keeps(&fm(json!({"status": "retired", "type": "guide"})), &pred));
    }
    #[test]
    fn a_bare_key_beside_an_or_is_hinted_at_and_not_refused() {
        // The precedence reading of `status=live|draft` is `status=live` or the
        // presence of a key `draft`, which is rarely meant and is still a real
        // thing to ask for. So it works, and it says so.
        let (pred, hints) = parse(&["status=live|draft"]);
        assert!(keeps(&fm(json!({"draft": true})), &pred), "the presence test has to keep working");
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("presence test for the key `draft`"), "{}", hints[0]);
        assert!(
            parse(&["status=live | status=draft"]).1.is_empty(),
            "an or of two comparisons needs no hint"
        );
    }
    #[test]
    fn the_predicate_that_emptied_a_result_is_named() {
        let rows = vec![(0, fm(json!({"status": "live"}))), (1, fm(json!({"status": "retired"})))];
        let pred = preds(&["status=live | status=draft", "type=guide"]);
        assert_eq!(
            blames(&rows, &pred),
            vec!["type=guide".to_string()],
            "only the predicate that kept nothing on its own is named"
        );
        assert!(blames(&rows, &preds(&["status=live"])).is_empty());
    }
    #[test]
    fn key_not_equal_still_passes_when_the_key_is_absent() {
        // The documented behaviour, kept: a filter must not silently drop the
        // documents nobody has annotated yet.
        assert!(keeps(&fm(json!({"title": "Beta"})), &preds(&["status!=deprecated"])));
    }
}

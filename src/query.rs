//! Answering a question: ranking, and assembling what is returned.
//!
//! Retrieval names candidates and the caller reads the file — D-01M1PP6HFHY7FW
//! — so what comes back is a range to read rather than a body. Several indexes
//! answer as one only when they are one space, adjacent sections that both rank
//! come back as one reference, and a row is checked against its file before it
//! is printed.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::config::{
    DEFAULT_ENDPOINT, budget, config_path, read_config_at, resolve_allow_insecure, resolve_api_key,
};
use crate::embed::{SAME_SPACE, dot, embed, same_space};
use crate::filters::{Pred, blames, keeps, parse_preds, pointed_at, scalar_text};
use crate::index::reindex;
use crate::sections::Section;
use crate::store::{self, Store};
use crate::{as_floats, map_vectors, open_index, stamp};

/// The rows a query keeps, best first, and how many the anti-join dropped.
/// One index a query reads, held open for as long as it ranks.
///
/// The map is given up before a refresh writes: a refresh re-indexes, and
/// D-01M1QHKG8KGB4Q chose a journal mode against concurrent access.
struct Opened {
    root: PathBuf,
    st: Store,
    meta: store::Meta,
    map: memmap2::Mmap,
}

/// Every index a query was pointed at, refusing what cannot be ranked together.
///
/// Dimension and budget are compared before the endpoint is asked anything.
/// Neither can be measured from a vector: the fingerprint is one short fixed
/// text embedded whole, so it reads the same at every budget, while a budget
/// divides a section too long for it into records at boundaries the budget
/// decides. Two indexes at different budgets hold different rows for the same
/// text and rank each on a different amount of it.
fn open_all(roots: &[PathBuf]) -> Result<Vec<Opened>> {
    // A root named twice is one root. Identical indexes rank identically, so
    // collapsing here is the same answer for one fewer scan.
    let mut canon: Vec<PathBuf> = Vec::new();
    for r in roots {
        let c = r.canonicalize().with_context(|| format!("{} cannot be read", r.display()))?;
        if !canon.contains(&c) {
            canon.push(c);
        }
    }

    let mut out: Vec<Opened> = Vec::new();
    for root in canon {
        let Some((st, meta)) = open_index(&root)? else {
            bail!("no index under {} — run `folio index` first", root.display());
        };
        let map = map_vectors(&root)?.context("the index has records but no vectors")?;
        out.push(Opened { root, st, meta, map });
    }

    let first = out.first().expect("a query names at least one root");
    for o in &out[1..] {
        if o.meta.dim != first.meta.dim {
            bail!(
                "the index under {} was built at dimension {} and the one under {} at {}, \
                 so their rows cannot be ranked against one question",
                first.root.display(),
                first.meta.dim,
                o.root.display(),
                o.meta.dim
            );
        }
        if o.meta.max_chars != first.meta.max_chars {
            bail!(
                "the index under {} was built at a budget of {} characters and the one under {} \
                 at {}, so the same section is divided differently in each and ranked on a \
                 different amount of its text — re-index one of them with \
                 `--max-chars {}`",
                first.root.display(),
                first.meta.max_chars,
                o.root.display(),
                o.meta.max_chars,
                first.meta.max_chars
            );
        }
    }
    // Where there is one index, an index built before folio recorded a
    // fingerprint is still answered: what that costs is an unverified answer
    // about the caller's own corpus. Where there are several it would join
    // proven indexes in one ordering, which is the failure the fingerprint
    // exists to prevent.
    if out.len() > 1
        && let Some(o) = out.iter().find(|o| o.meta.fingerprint.is_empty())
    {
        bail!(
            "the index under {} carries no fingerprint, so folio cannot show it holds the \
             same vectors as the others — run `folio index --rebuild` there",
            o.root.display()
        );
    }
    Ok(out)
}

/// The query's own vector, once every index has shown it is the same space.
///
/// Each index stores the text of its fingerprint beside the vector the endpoint
/// returned for it, and each of those texts rides in the query's own request:
/// folio batches 32 inputs and a query sends one, so proving a handful of
/// indexes costs no extra round trip. The endpoint asked is the first root's,
/// and that is enough for all of them — an index in the endpoint's space and a
/// second index in the endpoint's space are in each other's.
fn embed_and_prove(
    opened: &[Opened],
    text: &str,
    api_key: Option<&str>,
    allow_insecure: bool,
) -> Result<Vec<f32>> {
    let first = &opened.first().expect("a query names at least one root").meta;
    let mut inputs: Vec<String> = Vec::new();
    for o in opened {
        if !o.meta.fingerprint.is_empty() && !inputs.contains(&o.meta.fingerprint) {
            inputs.push(o.meta.fingerprint.clone());
        }
    }
    inputs.push(text.to_string());
    let mut vectors = embed(&first.endpoint, &first.model, api_key, allow_insecure, &inputs)?;
    let asked = vectors.pop().expect("one input yields one vector");

    // A query does not discard an index it was asked to read. Its own vector
    // comes from the weights answering now and every row comes from the weights
    // that built the index, so it cannot rank them against each other, and
    // re-indexing is the caller's decision.
    for o in opened {
        if o.meta.fingerprint.is_empty() {
            continue;
        }
        let at = inputs
            .iter()
            .position(|t| *t == o.meta.fingerprint)
            .expect("every fingerprint text was sent");
        let near = same_space(&vectors[at], &o.meta.fingerprint_vec);
        if near < SAME_SPACE {
            bail!(
                "the endpoint at {} no longer returns what the index under {} was built with \
                 (fingerprint {near:.6} against the one it carries, and {SAME_SPACE} is where \
                 the same weights sit), so those rows cannot be ranked against your question — \
                 run `folio index --rebuild` there, or point folio back at the endpoint that \
                 built it",
                first.endpoint,
                o.root.display()
            );
        }
    }
    Ok(asked)
}

/// Rank every index the query reads, as one list.
///
/// A slot names a row in one index's matrix, so a hit carries which index it
/// came from. Scores are comparable because `open_all` and `embed_and_prove`
/// have shown the indexes are one space, which is what lets one sort stand and
/// `--limit` keep its meaning.
/// A candidate row in an index's matrix, scored by dot product against the query.
#[derive(Clone, Copy)]
struct ScoredSlot {
    score: f32,
    owner: usize,
    slot: usize,
}

fn rank(
    opened: &[Opened],
    q: &[f32],
    preds: &[Pred],
    exclude_pointed_by: &[String],
    identity: &str,
) -> Result<(Vec<ScoredSlot>, usize)> {
    let mut hits: Vec<ScoredSlot> = Vec::new();
    let mut dropped = 0usize;

    // Frontmatter is read only when something decides on it. Without a filter
    // and without a join, ranking needs the vectors and the list of rows that
    // are still live, and nothing else: on 119,359 sections that is 10 ms of
    // slots against 180 ms of whole records.
    if preds.is_empty() && exclude_pointed_by.is_empty() {
        for (owner, o) in opened.iter().enumerate() {
            let floats = as_floats(&o.map)?;
            let dim = o.meta.dim;
            for slot in o.st.slots()? {
                hits.push(ScoredSlot {
                    score: dot(q, &floats[slot * dim..(slot + 1) * dim]),
                    owner,
                    slot,
                });
            }
        }
    } else {
        let mut rows: Vec<Vec<(usize, Map<String, Value>)>> = Vec::new();
        for o in opened {
            rows.push(o.st.slots_with_fm()?);
        }
        // The right side of the anti-join is every row of every index the
        // query reads. A pointer is a fact about a corpus and not about which
        // of its indexes holds it, so a record whose successor sits in another
        // of them is retired here too.
        let mut pointed = BTreeSet::new();
        for r in &rows {
            pointed.extend(pointed_at(r, exclude_pointed_by));
        }
        let superseded = |fm: &Map<String, Value>| {
            !pointed.is_empty()
                && fm.get(identity).is_some_and(|v| pointed.contains(&scalar_text(v)))
        };
        for (owner, o) in opened.iter().enumerate() {
            let floats = as_floats(&o.map)?;
            let dim = o.meta.dim;
            for (slot, fm) in &rows[owner] {
                if !keeps(fm, preds) {
                    continue;
                }
                if superseded(fm) {
                    dropped += 1;
                    continue;
                }
                hits.push(ScoredSlot {
                    score: dot(q, &floats[slot * dim..(slot + 1) * dim]),
                    owner,
                    slot: *slot,
                });
            }
        }
    }
    hits.sort_by(|a, b| b.score.total_cmp(&a.score));
    Ok((hits, dropped))
}

/// Which of `sections`' files no longer look the way the index recorded them.
///
/// Only the files behind the rows about to be returned, because those are the
/// ones a caller is about to open. A file that changed and did not surface
/// costs a candidate, which the index was always allowed to cost.
fn moved_since_indexed(st: &Store, root: &Path, sections: &[Section]) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    for path in sections.iter().map(|s| &s.path).collect::<BTreeSet<_>>() {
        let Some(was) = st.stamp_of(path)? else {
            out.insert(path.clone());
            continue;
        };
        match fs::metadata(root.join(path)) {
            Ok(m) => {
                let now = stamp(&m, was.hash);
                if now.len != was.len || now.mtime != was.mtime {
                    out.insert(path.clone());
                }
            }
            Err(_) => {
                out.insert(path.clone());
            }
        }
    }
    Ok(out)
}

/// How a reference prints.
///
/// One root that is the working directory prints the path the caller already
/// knows the root of. Anything else prints which root the section came from,
/// because two indexes may hold the same relative path and `--paths-only`
/// exists for a caller that opens what it is handed. Relative to the working
/// directory where it can be, so that the common case reads like the one-root
/// case.
///
/// A root above the working directory is where that matters: a query asked
/// from inside a corpus is answered from the corpus, and a bare relative path
/// would then name a file from a directory the caller is not in.
fn reference(roots: &[PathBuf], owner: usize, rel: &str) -> String {
    let cwd = std::env::current_dir().ok();
    if roots.len() == 1 && cwd.as_deref() == Some(roots[0].as_path()) {
        return rel.to_string();
    }
    let joined = roots[owner].join(rel);
    cwd.and_then(|cwd| joined.strip_prefix(cwd).ok().map(|p| p.display().to_string()))
        .unwrap_or_else(|| joined.display().to_string())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_query(
    roots: &[PathBuf],
    text: &str,
    wheres: &[String],
    exclude_pointed_by: &[String],
    identity: &str,
    limit: usize,
    paths_only: bool,
    refresh: bool,
    allow_insecure: bool,
) -> Result<()> {
    let user = read_config_at(&config_path())?;
    let allow_insecure = resolve_allow_insecure(allow_insecure, user.allow_insecure);
    let (preds, hints) = parse_preds(wheres)?;
    // Said once, before any work: a hint about what was typed does not depend
    // on what the index holds.
    for hint in &hints {
        eprintln!("{hint}");
    }
    let mut q: Option<Vec<f32>> = None;
    // At most one refresh. A result still stale after re-indexing means the
    // files are moving while folio reads them, and saying so beats looping.
    let mut refreshed = false;

    loop {
        let opened = open_all(roots)?;
        let named: Vec<PathBuf> = opened.iter().map(|o| o.root.clone()).collect();
        let query_endpoint =
            opened.first().map(|o| o.meta.endpoint.as_str()).unwrap_or(DEFAULT_ENDPOINT);
        let (api_key, _) = resolve_api_key(query_endpoint, user.api_key.as_deref());
        if q.is_none() {
            q = Some(embed_and_prove(&opened, text, api_key.as_deref(), allow_insecure)?);
        }
        let q = q.as_deref().expect("embedded above");

        let (hits, dropped) = rank(&opened, q, &preds, exclude_pointed_by, identity)?;
        // Hydrated wider than the limit, because merging joins pieces that both
        // rank and the limit counts merged rows rather than sections.
        // Never below the limit: a caller asking for every record must be able
        // to receive every record, which D-01M1PP6HGWJMQC's fence checks.
        let window = (limit * MERGE_WINDOW).max(32);
        let top: Vec<ScoredSlot> = hits.iter().take(window).copied().collect();

        // Hydrated one index at a time and put back in the order they ranked,
        // because a slot means a row in one index's matrix.
        let mut hydrated: Vec<Option<Section>> = vec![None; top.len()];
        for (owner, o) in opened.iter().enumerate() {
            let mut at: Vec<usize> = Vec::new();
            let mut want: Vec<usize> = Vec::new();
            for (i, hit) in top.iter().enumerate() {
                if hit.owner == owner {
                    at.push(i);
                    want.push(hit.slot);
                }
            }
            if want.is_empty() {
                continue;
            }
            for (i, sec) in at.into_iter().zip(o.st.hydrate(&want)?) {
                hydrated[i] = Some(sec);
            }
        }

        // A section two indexes hold is one section. Joined to their roots both
        // copies name one file, and one space at one budget divides that file
        // the same way, so their ranges coincide rather than nearly coincide.
        // Which copy is kept is not observable: both print the same reference,
        // and the scores are a batch position apart, which is 1.1e-7 where
        // three decimals are printed.
        let mut rows: Vec<Section> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
        let mut scores: Vec<f32> = Vec::new();
        let mut seen: HashSet<(PathBuf, usize, usize)> = HashSet::new();
        let mut held = 0usize;
        for (i, hit) in top.iter().enumerate() {
            let sec = hydrated[i].take().expect("every ranked slot hydrated");
            let key = (named[hit.owner].join(&sec.path), sec.start, sec.end);
            if !seen.insert(key) {
                held += 1;
                continue;
            }
            rows.push(sec);
            owners.push(hit.owner);
            scores.push(hit.score);
        }

        let mut merged = merge_adjacent(&rows, &owners, &scores);
        merged.truncate(limit);

        // Always asked, whatever `refresh` says: one stat per returned row is
        // free, and a caller told to answer from the index as it stands is the
        // one who most needs to know where it does not.
        let mut returned: Vec<Vec<Section>> = vec![Vec::new(); opened.len()];
        for h in &merged {
            for &i in &h.pieces {
                returned[owners[i]].push(rows[i].clone());
            }
        }
        let mut stale: Vec<HashSet<String>> = Vec::with_capacity(opened.len());
        for (owner, o) in opened.iter().enumerate() {
            stale.push(moved_since_indexed(&o.st, &o.root, &returned[owner])?);
        }
        let stale_count: usize = stale.iter().map(HashSet::len).sum();

        // What a refresh needs, taken before the maps are given up. Only the
        // indexes with a stale row are re-indexed: the others answered from
        // files that have not moved.
        let plans: Vec<(PathBuf, String, String, usize)> = if refresh && !refreshed {
            opened
                .iter()
                .enumerate()
                .filter(|(owner, _)| !stale[*owner].is_empty())
                .map(|(_, o)| {
                    (
                        o.root.clone(),
                        o.meta.endpoint.clone(),
                        o.meta.model.clone(),
                        budget(o.meta.max_chars),
                    )
                })
                .collect()
        } else {
            Vec::new()
        };
        drop(opened);

        if !plans.is_empty() {
            let mut reembedded = 0usize;
            let mut ran = false;
            for (root, endpoint, model, budget) in &plans {
                // Declined rather than waited: another folio is already
                // writing, and its result will be at least as fresh as this
                // one's would be.
                if let Some(r) = reindex(
                    root,
                    endpoint,
                    model,
                    api_key.as_deref(),
                    allow_insecure,
                    *budget,
                    false,
                    std::time::Duration::ZERO,
                )? {
                    reembedded += r.reembedded;
                    ran = true;
                }
            }
            if ran {
                let notice = format!(
                    "({stale_count} of the files behind this result had changed; {reembedded} file(s) re-embedded before answering)"
                );
                if paths_only {
                    eprintln!("{notice}")
                } else {
                    println!("{notice}")
                }
                refreshed = true;
                continue;
            }
        }

        // Reported before the empty case, so that "nothing matched" is never the
        // only thing a caller hears when the anti-join is what emptied the result.
        if dropped > 0 {
            let notice = format!(
                "({dropped} section(s) dropped as pointed at by {})",
                exclude_pointed_by.join(", ")
            );
            if paths_only { eprintln!("{notice}") } else { println!("{notice}") }
        }
        // Said even though the answer is the same either way, because nothing
        // else is in a position to: an overlap costs one embedding per index
        // covering the file on every edit, and a parent's walk skips a child's
        // `.folio/` as hidden and never learns the child index exists.
        if held > 0 {
            let notice = format!(
                "({held} section(s) held by more than one of these indexes, returned once)"
            );
            if paths_only { eprintln!("{notice}") } else { println!("{notice}") }
        }
        if merged.is_empty() {
            let mut said = vec!["no section passed the filter".to_string()];
            // Read again rather than carried through the ranking, because this
            // is the path where the answer was empty: one more pass over the
            // records costs a query that returned nothing anyway.
            if !preds.is_empty() {
                // Opened again rather than held: the maps were given up above,
                // and this path is reached only when the answer was empty.
                let mut all: Vec<(usize, Map<String, Value>)> = Vec::new();
                for root in &named {
                    all.extend(Store::open(root)?.slots_with_fm()?);
                }
                for text in blames(&all, &preds) {
                    said.push(format!("  {text} matched none on its own"));
                }
            }
            let said = said.join("\n");
            if paths_only {
                eprintln!("{said}")
            } else {
                println!("{said}")
            }
            return Ok(());
        }
        for (i, h) in merged.iter().enumerate() {
            // A row still stale here is one the refresh could not take, so the
            // line range may have moved. Said out loud rather than left to the
            // caller to discover by reading the wrong lines.
            let is_stale = stale[h.owner].contains(&h.path);
            let at = reference(&named, h.owner, &h.path);
            if paths_only {
                if is_stale {
                    eprintln!("{at}:{}-{} has moved since it was indexed", h.start, h.end);
                }
                println!("{at}:{}-{}", h.start, h.end);
                continue;
            }
            let mark = if is_stale { "  (stale)" } else { "" };
            let joined = match h.pieces.len() {
                1 => String::new(),
                n => format!("  ({n} sections)"),
            };
            println!("#{}  {:.3}  {at}:{}-{}{mark}", i + 1, h.score, h.start, h.end);
            println!("        {}{joined}", h.trail);
        }
        return Ok(());
    }
}

/// How many ranked sections are hydrated before merging.
///
/// Merging joins pieces that both rank, so a piece below this window joins
/// nothing and a run stops there. Hydrating every record instead would restore
/// the read D-01M1QWB73G6WNZ removed, which was 180 ms of a 267 ms query.
const MERGE_WINDOW: usize = 8;
/// One range to read, and the ranked sections it was assembled from.
struct Hit {
    /// Which of the indexes read holds this, so that the reference prints the
    /// root it came from and the staleness check asks the right store.
    owner: usize,
    path: String,
    start: usize,
    end: usize,
    score: f32,
    trail: String,
    pieces: Vec<usize>,
}

/// Join ranked sections that touch, so that a result names one range to read.
///
/// Five rows covering lines 3-6, 7-10, 11-14, 15-18 and 19-22 of one file
/// describe one read of lines 3-22 and leave the caller to work that out.
///
/// The merged score is the mean weighted by lines, so it reads as relevance per
/// line read: a tight pointer outranks a broad one that contains it. Lines
/// rather than characters because a query knows line ranges and never reads the
/// text.
///
/// Only pieces that touch are joined. If two sections rank and the ones between
/// them do not, the covering range would be mostly text that nothing matched.
///
/// Touching is not enough on its own. A section that answers the question sits
/// next to one that merely surrounds it, and joining those two replaces a tight
/// pointer with a loose one: measured on a fixture, a three-line answer at 0.766
/// became a seven-line range at 0.618. So a piece joins a run only while it is
/// comparably relevant to it, and a background neighbour is left out.
fn merge_adjacent(rows: &[Section], owners: &[usize], scores: &[f32]) -> Vec<Hit> {
    // Keyed by the index as well as the path, because two indexes may hold the
    // same relative path and two files are not one run.
    let mut by_path: BTreeMap<(usize, &str), Vec<usize>> = BTreeMap::new();
    for (i, s) in rows.iter().enumerate() {
        by_path.entry((owners[i], &s.path)).or_default().push(i);
    }

    let mut out: Vec<Hit> = Vec::new();
    for ((owner, path), mut idx) in by_path {
        idx.sort_by_key(|&i| rows[i].start);
        let mut run: Vec<usize> = Vec::new();
        let mut best = 0.0f32;
        for i in idx {
            let touches = run.last().is_some_and(|&last| rows[last].end + 1 == rows[i].start);
            // ponytail: one ratio, read off a fixture where 0.741 against 0.780
            // is one answer in five pieces and 0.507 against 0.766 is background
            // beside an answer. A corpus should set it, and cosine scales differ
            // by model, so this is a starting point rather than a constant.
            let alike = {
                let (lo, hi) = if scores[i] < best { (scores[i], best) } else { (best, scores[i]) };
                lo >= 0.9 * hi
            };
            if !run.is_empty() && !(touches && alike) {
                out.push(assemble(owner, path, rows, scores, &run));
                run.clear();
                best = 0.0;
            }
            best = best.max(scores[i]);
            run.push(i);
        }
        if !run.is_empty() {
            out.push(assemble(owner, path, rows, scores, &run));
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    out
}

fn assemble(owner: usize, path: &str, rows: &[Section], scores: &[f32], run: &[usize]) -> Hit {
    let start = rows[run[0]].start;
    let end = rows[run[run.len() - 1]].end;
    let lines = |i: usize| (rows[i].end + 1 - rows[i].start) as f32;
    let total: f32 = run.iter().map(|&i| lines(i)).sum();
    let score = run.iter().map(|&i| scores[i] * lines(i)).sum::<f32>() / total;
    Hit {
        owner,
        path: path.to_string(),
        start,
        end,
        score,
        trail: shared_trail(rows, run),
        pieces: run.to_vec(),
    }
}

/// The trail a merged range is filed under: what its pieces have in common.
///
/// Four notes under one heading are that heading. When the pieces share
/// nothing, the first one's trail is printed, because a reader following a
/// range reads from its start.
fn shared_trail(rows: &[Section], run: &[usize]) -> String {
    let trails: Vec<Vec<String>> = run
        .iter()
        .map(|&i| {
            let mut t = rows[i].breadcrumb.clone();
            t.push(rows[i].heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
            t
        })
        .collect();
    let mut common: Vec<String> = trails[0].clone();
    for t in &trails[1..] {
        let keep = common.iter().zip(t).take_while(|(a, b)| a == b).count();
        common.truncate(keep);
    }
    if common.is_empty() {
        return trails[0].join(" > ");
    }
    common.join(" > ")
}

/// The heading trail under a result, as it is printed beneath the reference.
pub(crate) fn trail_of(s: &Section) -> String {
    let mut trail = s.breadcrumb.clone();
    trail.push(s.heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
    trail.join(" > ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sec(path: &str, start: usize, end: usize) -> Section {
        Section {
            path: path.to_string(),
            start,
            end,
            heading: Some("H".to_string()),
            breadcrumb: vec!["H".to_string()],
            fm: Map::new(),
            truncated: false,
            text: String::new(),
        }
    }
    #[test]
    fn one_path_in_two_indexes_is_two_runs() {
        let rows = [sec("a.md", 1, 4), sec("a.md", 5, 8)];
        let one = merge_adjacent(&rows, &[0, 0], &[0.8, 0.79]);
        assert_eq!(one.len(), 1, "touching pieces of one file are one run");
        let two = merge_adjacent(&rows, &[0, 1], &[0.8, 0.79]);
        assert_eq!(two.len(), 2, "one relative path in two indexes is two files");
    }
    #[test]
    fn a_reference_is_bare_only_from_the_root_it_names() {
        let here = std::env::current_dir().expect("a working directory");
        let one = [here];
        assert_eq!(reference(&one, 0, "docs/a.md"), "docs/a.md");

        // A corpus found by walking upward is not the working directory, and a
        // bare path would name a file from a directory the caller is not in.
        let above = [PathBuf::from("/corpus")];
        assert_eq!(reference(&above, 0, "docs/a.md"), "/corpus/docs/a.md");

        let two = [PathBuf::from("/corpus"), PathBuf::from("/other")];
        assert_eq!(reference(&two, 1, "docs/a.md"), "/other/docs/a.md");
    }
}

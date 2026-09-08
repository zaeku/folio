//! Keeping an index current: what changed, what moved, and what is embedded
//! again.
//!
//! The unit is the file — D-01M1PP6HM6WGT4 — because a stamp says the index is
//! current for one, so a run commits whole files and one that dies keeps the
//! ones it finished. A file that only moved keeps the vectors it had, and the
//! index is written in proportion to what changed.

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::Hasher;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

use crate::config::{
    DEFAULT_ENDPOINT, DEFAULT_MODEL, PROJECT_CONFIG, config_path, enclosing_index,
    nearest_declaration, read_config_at, resolve, resolve_allow_insecure, resolve_api_key,
};
use crate::embed::{SAME_SPACE, calibrate, embed, same_space};
use crate::sections::{self, Section};
use crate::store::{self, Meta, Stamp, Store};
use crate::{as_floats, map_vectors, stamp};

/// The text whose embedding says which weights answered. Its value does not
/// matter; that every run sends the same one does. An index stores the text it
/// was built with, so changing this affects indexes built after it and no other.
pub(crate) const FINGERPRINT: &str = "folio probes the endpoint to learn which weights answered.";
pub(crate) fn hash(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(bytes);
    h.finish()
}
/// What a re-index did, for whoever asked for it to say so.
pub(crate) struct Indexed {
    pub(crate) files: usize,
    pub(crate) sections: usize,
    pub(crate) truncated: usize,
    /// The budget these sections were cut at, which is `--max-chars` unless the
    /// endpoint refused it.
    pub(crate) budget: usize,
    pub(crate) dim: usize,
    pub(crate) reembedded: usize,
    pub(crate) moved: usize,
    pub(crate) retired: usize,
    pub(crate) dead: usize,
    pub(crate) compacted: usize,
}
/// Pair each path that is gone with a path that arrived carrying the same
/// content, as `(from, to)`.
///
/// A hash still held by a file that is present is a copy and not a move, so
/// only paths absent from `current` give their records away, and each gives
/// them to one arrival. Both sides are taken in sorted order, so a rename of
/// several files with identical content pairs the same way every run.
fn pair_moves(
    prev: &HashMap<String, Stamp>,
    current: &HashMap<String, Stamp>,
    changed: &HashSet<String>,
) -> Vec<(String, String)> {
    let mut gone: HashMap<u64, Vec<&String>> = HashMap::new();
    for (path, st) in prev {
        if !current.contains_key(path) {
            gone.entry(st.hash).or_default().push(path);
        }
    }
    for paths in gone.values_mut() {
        paths.sort_by(|a, b| b.cmp(a));
    }
    let mut arrived: Vec<&String> = changed.iter().filter(|p| !prev.contains_key(*p)).collect();
    arrived.sort();
    let mut moves = Vec::new();
    for to in arrived {
        let Some(from) = gone.get_mut(&current[to].hash).and_then(|paths| paths.pop()) else {
            continue;
        };
        moves.push((from.clone(), to.clone()));
    }
    moves
}
/// Bring the index under `root` up to date with the files under it.
///
/// `Ok(None)` means another process holds the write lock and this one declined
/// to wait: a re-index nobody asked for is not worth blocking on, and one the
/// user asked for says so rather than hanging.
pub(crate) fn reindex(
    root: &Path,
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    allow_insecure: bool,
    max_chars: usize,
    rebuild: bool,
    wait: std::time::Duration,
) -> Result<Option<Indexed>> {
    let root = root.canonicalize().with_context(|| format!("{} not found", root.display()))?;
    if max_chars == 0 {
        bail!("a section budget of 0 characters would embed nothing");
    }
    let st = Store::open(&root)?;
    if st.compacting()? {
        bail!(
            "a compaction of {} did not finish, so the records may name the wrong rows — rerun with --rebuild",
            store::vectors_path(&root).display()
        );
    }
    // Everything below reads the row the matrix has grown to and then writes
    // there, so it is one update and it is taken as one.
    if !st.lock(wait)? {
        return Ok(None);
    }
    let prev = st.meta()?;
    // A vector space is what the endpoint returns, not what it is called: a
    // server restarted on other weights keeps the URL and the model string it
    // had. So the index is kept or discarded on what answers now, and the probe
    // sent is the one this index was built with rather than today's default.
    let fingerprint_text =
        if prev.fingerprint.is_empty() { FINGERPRINT } else { prev.fingerprint.as_str() }
            .to_string();
    let mut fingerprint: Vec<f32> = Vec::new();
    let mut drifted: Option<f32> = None;
    if !rebuild && prev.dim > 0 {
        fingerprint = embed(endpoint, model, api_key, allow_insecure, &[fingerprint_text.clone()])?
            .pop()
            .expect("one input yields one vector");
        if !prev.fingerprint_vec.is_empty() {
            let near = same_space(&fingerprint, &prev.fingerprint_vec);
            if near < SAME_SPACE {
                drifted = Some(near);
            }
        }
    }
    let reuse = !rebuild && prev.dim > 0 && drifted.is_none();
    if let Some(near) = drifted {
        // Discarding is the decision working. Saying so, with the number that
        // caused it, is the difference between that and a re-index someone
        // cannot account for — and it separates a model someone swapped from a
        // runtime someone upgraded.
        println!(
            "the endpoint at {endpoint} no longer returns what this index was built with \
             (fingerprint {near:.6} against the one it carries, and {SAME_SPACE} is where the \
             same weights sit) — \
             re-embedding every section"
        );
    }
    let prev_files = if reuse { st.files()? } else { HashMap::new() };
    if !reuse {
        st.clear()?;
        fs::write(store::vectors_path(&root), [])?;
    }
    // The walk is threaded because it dominated what was left of a re-index:
    // on 14,616 files it ran 449-804 ms in one thread and 172 ms across ten.
    // Results land in whatever order the threads finish, which is fine because
    // they are collected into maps and a set.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let found: Mutex<Vec<(String, Stamp, bool)>> = Mutex::new(Vec::new());
    let failed: Mutex<Vec<String>> = Mutex::new(Vec::new());

    ignore::WalkBuilder::new(&root).threads(threads).build_parallel().run(|| {
        Box::new(|entry| {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    failed.lock().unwrap().push(e.to_string());
                    return ignore::WalkState::Continue;
                }
            };
            if !entry.file_type().is_some_and(|t| t.is_file()) {
                return ignore::WalkState::Continue;
            }
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("md") {
                return ignore::WalkState::Continue;
            }
            let Ok(suffix) = path.strip_prefix(&root) else {
                return ignore::WalkState::Continue;
            };
            let rel = suffix.to_string_lossy().replace('\\', "/");

            // Listing a file is cheap and reading it is not: stamping all
            // 14,616 costs 33 ms against 1294 ms to read and hash them. So a
            // file whose length and modification time are what the index
            // recorded keeps the hash recorded beside them and is never
            // opened. Every other file is read, and its content decides.
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(e) => {
                    failed.lock().unwrap().push(format!("{rel}: {e}"));
                    return ignore::WalkState::Continue;
                }
            };
            let was = if reuse { prev_files.get(&rel) } else { None };
            let unmoved = was.is_some_and(|w| {
                let s = stamp(&meta, w.hash);
                s.len == w.len && s.mtime == w.mtime
            });
            let h = if unmoved {
                was.expect("unmoved implies a recorded stamp").hash
            } else {
                match fs::read(path) {
                    Ok(bytes) => hash(&bytes),
                    Err(e) => {
                        failed.lock().unwrap().push(format!("{rel}: {e}"));
                        return ignore::WalkState::Continue;
                    }
                }
            };
            let moved = was.is_none_or(|w| w.hash != h);
            found.lock().unwrap().push((rel, stamp(&meta, h), moved));
            ignore::WalkState::Continue
        })
    });

    let failed = failed.into_inner().unwrap();
    if let Some(first) = failed.first() {
        bail!("{} file(s) could not be read, starting with {first}", failed.len());
    }

    let mut current: HashMap<String, Stamp> = HashMap::new();
    let mut changed: HashSet<String> = HashSet::new();
    for (rel, st, moved) in found.into_inner().unwrap() {
        if moved {
            changed.insert(rel.clone());
        }
        current.insert(rel, st);
    }

    // A file that changed and a file that is gone retire their records the
    // same way; only the first also contributes new ones.
    let mut retired_paths: HashSet<String> = changed.clone();
    retired_paths.extend(prev_files.keys().filter(|p| !current.contains_key(*p)).cloned());

    // A file that only moved carries the content its records already describe,
    // so those records take the new path and nothing is embedded. Identity is
    // the path, so without this a rename reads as a delete and a create.
    let moves = pair_moves(&prev_files, &current, &changed);
    for (from, to) in &moves {
        retired_paths.remove(from);
        retired_paths.remove(to);
        changed.remove(to);
    }

    let mut fresh: Vec<Section> = Vec::new();
    for rel in &changed {
        let source = fs::read_to_string(root.join(rel))?;
        for s in sections::split(rel, &source) {
            fresh.extend(sections::to_budget(s, max_chars));
        }
    }

    // The budget is checked against the server before anything is embedded with
    // it. A refusal partway through would abandon the run, and a run that
    // abandons at section 90,000 has spent half an hour to say what one request
    // says here.
    let mut max_chars = max_chars;
    if let Some(longest) = fresh.iter().max_by_key(|s| s.text.chars().count()) {
        let longest = longest.text.clone();
        let fitted = calibrate(endpoint, model, api_key, allow_insecure, &longest, max_chars)?;
        if fitted < max_chars {
            println!("  budget for this run: {fitted} characters, not {max_chars}");
            max_chars = fitted;
            fresh = fresh.into_iter().flat_map(|s| sections::to_budget(s, max_chars)).collect();
        }
    }

    // The fingerprint this run leaves behind, resolved before anything is
    // committed. Its length is the dimension, so one request settles both — and
    // both have to be in the meta from the first batch on. Without them a run
    // that dies leaves rows belonging to no space, and the next run discards
    // every one of them instead of continuing.
    let (fingerprint_text, fingerprint) = if fresh.is_empty() {
        (prev.fingerprint.clone(), prev.fingerprint_vec.clone())
    } else if reuse {
        (fingerprint_text, fingerprint)
    } else {
        let v = embed(endpoint, model, api_key, allow_insecure, &[FINGERPRINT.to_string()])?
            .pop()
            .expect("one input yields one vector");
        (FINGERPRINT.to_string(), v)
    };
    let mut dim = if reuse { prev.dim } else { fingerprint.len() };

    let mut retired = 0;
    // A file that vanished loses its records now: there is nothing to embed for
    // it, so nothing later in this run can conflict with the deletion, and a
    // run that dies after it has still done the right thing.
    let vanished: HashSet<String> = retired_paths.difference(&changed).cloned().collect();
    st.rename(&moves)?;
    retired += st.retire(&vanished)?;
    st.forget(&vanished)?;
    st.stamp(moves.iter().filter_map(|(_, to)| current.get_key_value(to)))?;
    st.set_meta(&Meta {
        model: model.to_string(),
        endpoint: endpoint.to_string(),
        dim,
        max_chars,
        fingerprint: fingerprint_text,
        fingerprint_vec: fingerprint,
    })?;
    st.unlock()?;

    // Committed in batches of whole files. A run that dies keeps the files it
    // finished, and the unit is the file because a stamp says the index is
    // current for one: written for a file whose sections are only half in, the
    // next run would skip the other half forever.
    for batch in batches(fresh, FLUSH) {
        let inputs: Vec<String> = batch.iter().map(|s| s.text.clone()).collect();
        let vectors = embed(endpoint, model, api_key, allow_insecure, &inputs)?;
        if dim == 0 {
            dim = vectors[0].len();
        }
        if let Some((i, v)) = vectors.iter().enumerate().find(|(_, v)| v.len() != dim) {
            bail!("{} came back at dim {} while the index is dim {dim}", batch[i].path, v.len());
        }
        // The lock spans reading the row the matrix has grown to, appending
        // there, and recording what was written — the same span it always had,
        // taken once per batch rather than once per run.
        if !st.lock(wait.max(std::time::Duration::from_secs(30)))? {
            bail!("another folio took the index while this one was embedding");
        }
        let base = st.high_water()?;
        append(&root, base, dim, &vectors)?;
        let touched: HashSet<String> = batch.iter().map(|s| s.path.clone()).collect();
        retired += st.retire(&touched)?;
        let placed: Vec<(usize, Section)> = (base..).zip(batch).collect();
        st.add(&placed)?;
        st.stamp(touched.iter().filter_map(|p| current.get_key_value(p)))?;
        st.unlock()?;
    }

    if !st.lock(wait.max(std::time::Duration::from_secs(30)))? {
        bail!("another folio took the index before this one could record what it did");
    }

    // Every stamp, not only the batches': a file someone touched without
    // changing keeps its content hash and needs its new length and time
    // recorded, or it is read again on every index from here on.
    st.stamp(current.iter())?;

    // Read inside the lock, so these describe what was just written and not
    // what someone else wrote next.
    let counts = st.counts()?;
    let dead = rows_in_matrix(&root, dim)?.saturating_sub(counts.sections);

    // A dead row costs a query nothing — it is skipped, not scanned — and costs
    // the disk its bytes. Reclaiming them is the one full rewrite folio makes,
    // so it waits until they outnumber the rows they sit beside. The flag goes
    // up in this commit: from here until the rewrite finishes, every command
    // refuses the index, which is what stands in for holding the lock across a
    // rename that no transaction can cover.
    let due = dead > counts.sections && dead > 0;
    if due {
        st.start_compacting()?;
    }
    st.unlock()?;

    let compacted = if due {
        let live = st.slots()?;
        compact(&root, &st, dim, &live)?
    } else {
        0
    };

    Ok(Some(Indexed {
        files: counts.files,
        sections: counts.sections,
        truncated: counts.truncated,
        budget: max_chars,
        dim,
        reembedded: changed.len(),
        moved: moves.len(),
        retired,
        dead,
        compacted,
    }))
}
pub(crate) fn cmd_index(
    root: &Path,
    endpoint: Option<&str>,
    model: Option<&str>,
    allow_insecure: bool,
    max_chars: usize,
    rebuild: bool,
) -> Result<()> {
    let declared = nearest_declaration(root).unwrap_or_else(|| root.join(PROJECT_CONFIG));
    let proj = read_config_at(&declared)?;
    let user = read_config_at(&config_path())?;
    // An overlap costs an embedding in each index on every edit of a file both
    // hold, and this is the only moment it is visible: a parent's walk skips a
    // child's `.folio/` as a hidden directory, so no later run learns of it.
    if let Some(above) = enclosing_index(root) {
        eprintln!(
            "  this is inside the index at {}, which also holds these files",
            above.display()
        );
    }
    let (endpoint, _) = resolve(
        endpoint,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (model, _) =
        resolve(model, "FOLIO_MODEL", proj.model.as_deref(), user.model.as_deref(), DEFAULT_MODEL);
    let (api_key, _) = resolve_api_key(&endpoint, user.api_key.as_deref());
    let allow_insecure = resolve_allow_insecure(allow_insecure, user.allow_insecure);
    let (endpoint, model) = (endpoint.as_str(), model.as_str());
    // An index the user asked for waits a little for one already running, and
    // then says who it is waiting for rather than hanging on it.
    let wait = std::time::Duration::from_secs(10);
    let Some(r) = reindex(
        root,
        endpoint,
        model,
        api_key.as_deref(),
        allow_insecure,
        max_chars,
        rebuild,
        wait,
    )?
    else {
        bail!(
            "another folio is writing the index under {} — try again once it is done",
            root.display()
        );
    };
    println!("indexed {} files · {} sections · dim {}", r.files, r.sections, r.dim);
    println!(
        "  {} re-embedded, {} replaced or removed, {} dead row(s)",
        r.reembedded, r.retired, r.dead
    );
    if r.moved > 0 {
        println!("  {} moved, keeping the vectors they had", r.moved);
    }
    if r.truncated > 0 {
        println!("  {} sections truncated at {} characters", r.truncated, r.budget);
    }
    if r.compacted > 0 {
        println!("  compacted: {} dead row(s) reclaimed", r.compacted);
    }
    Ok(())
}
/// Split sections into batches of at least `want`, never cutting a file in two.
///
/// `fresh` is built one file at a time, so a file's sections are already
/// together; this only chooses where to end a batch. A single file larger than
/// `want` becomes a batch of its own rather than being divided.
fn batches(fresh: Vec<Section>, want: usize) -> Vec<Vec<Section>> {
    let mut out: Vec<Vec<Section>> = Vec::new();
    let mut batch: Vec<Section> = Vec::new();
    for s in fresh {
        let boundary = batch.last().is_some_and(|last| last.path != s.path);
        if boundary && batch.len() >= want {
            out.push(std::mem::take(&mut batch));
        }
        batch.push(s);
    }
    if !batch.is_empty() {
        out.push(batch);
    }
    out
}
/// Rows the matrix holds, live and dead.
fn rows_in_matrix(root: &Path, dim: usize) -> Result<usize> {
    if dim == 0 {
        return Ok(0);
    }
    let vecs = store::vectors_path(root);
    Ok(if vecs.exists() { fs::metadata(&vecs)?.len() as usize / (dim * 4) } else { 0 })
}
/// Write `vectors` at row `base`, leaving every row before it untouched. The
/// file is not truncated: rows past the end of the write are dead, and a write
/// that fails must not shorten the matrix under the records that name them.
pub(crate) fn append(root: &Path, base: usize, dim: usize, vectors: &[Vec<f32>]) -> Result<()> {
    use std::io::{Seek, SeekFrom};
    let path = store::vectors_path(root);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("{} could not be written", path.display()))?;
    file.seek(SeekFrom::Start((base * dim * 4) as u64))?;
    let mut w = BufWriter::new(&mut file);
    for v in vectors {
        for x in v {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()?;
    drop(w);
    file.sync_all()?;
    Ok(())
}
/// Copy the live rows to the front of a new matrix and renumber the records to
/// match. The two land in separate steps, so a flag is set across them: an
/// index found mid-compaction says so rather than ranking against wrong rows.
fn compact(root: &Path, st: &Store, dim: usize, live: &[usize]) -> Result<usize> {
    let before = rows_in_matrix(root, dim)?;
    let path = store::vectors_path(root);
    let tmp = path.with_extension("f32.compacting");
    {
        let map = map_vectors(root)?.context("there is no matrix to compact")?;
        let floats = as_floats(&map)?;
        let mut w = BufWriter::new(fs::File::create(&tmp)?);
        for slot in live {
            for x in &floats[slot * dim..(slot + 1) * dim] {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.flush()?;
        w.into_inner()?.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    // Nobody else can be writing — they all refuse an index carrying the flag —
    // so this waits for the lock rather than declining it.
    if !st.lock(std::time::Duration::from_secs(30))? {
        bail!("could not take the write lock to renumber after compacting");
    }
    st.renumber(live)?;
    st.unlock()?;
    Ok(before - live.len())
}
/// How many sections an index commits at a time. A run that dies keeps every
/// batch before the one it was in, so this is what a kill costs: at the 0.53 s
/// per 32 inputs measured on 2026-09-06, about eight seconds of embedding.
pub(crate) const FLUSH: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;

    fn stamped(hash: u64) -> Stamp {
        Stamp { hash, len: 1, mtime: 0 }
    }
    #[test]
    fn a_renamed_file_hands_its_records_to_the_new_path() {
        let prev = HashMap::from([("a.md".to_string(), stamped(7))]);
        let current = HashMap::from([("b.md".to_string(), stamped(7))]);
        let changed = HashSet::from(["b.md".to_string()]);
        assert_eq!(
            pair_moves(&prev, &current, &changed),
            vec![("a.md".to_string(), "b.md".to_string())]
        );
    }
    #[test]
    fn a_copy_takes_no_records_from_the_file_it_copied() {
        let prev = HashMap::from([("a.md".to_string(), stamped(7))]);
        let current =
            HashMap::from([("a.md".to_string(), stamped(7)), ("b.md".to_string(), stamped(7))]);
        let changed = HashSet::from(["b.md".to_string()]);
        assert!(pair_moves(&prev, &current, &changed).is_empty());
    }
    #[test]
    fn one_departure_supplies_only_one_of_two_arrivals() {
        let prev = HashMap::from([("a.md".to_string(), stamped(7))]);
        let current =
            HashMap::from([("b.md".to_string(), stamped(7)), ("c.md".to_string(), stamped(7))]);
        let changed = HashSet::from(["b.md".to_string(), "c.md".to_string()]);
        let moves = pair_moves(&prev, &current, &changed);
        assert_eq!(moves, vec![("a.md".to_string(), "b.md".to_string())]);
    }
}

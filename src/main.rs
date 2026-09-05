//! folio — section-scoped semantic search over a markdown corpus.
//!
//! The index holds references (path, line range, frontmatter), never bodies.
//! Retrieval names sections to read; reading the file is a separate step, so a
//! stale index costs a wasted candidate and never a wrong quotation.

mod sections;
mod store;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use sections::Section;
use serde::{Deserialize, Serialize};
use store::{Meta, Stamp, Store};
use serde_json::{Map, Value};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::hash::{DefaultHasher, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

const BATCH: usize = 32;

#[derive(Parser)]
#[command(name = "folio", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Index every markdown file under `root`, re-embedding only changed files.
    Index {
        #[arg(default_value = ".")]
        root: PathBuf,
        /// Any OpenAI-compatible embeddings endpoint.
        #[arg(
            long,
            env = "FOLIO_ENDPOINT",
            default_value = "http://127.0.0.1:8080/v1/embeddings"
        )]
        endpoint: String,
        #[arg(long, env = "FOLIO_MODEL", default_value = "default")]
        model: String,
        /// Cap on the text sent per section. Exceeding sections are marked.
        ///
        /// A character budget standing in for the model's token limit, which
        /// folio cannot see. It cannot bound a token count in general — a
        /// byte-level tokenizer can spend more than one token on a multi-byte
        /// character — so this default is set below the 8192-token context of
        /// the models folio is measured on rather than at a round number. Raise
        /// it for a longer-context model, and read the truncated count.
        #[arg(long, default_value_t = 8_000)]
        max_chars: usize,
        /// Discard the existing index instead of updating it.
        #[arg(long)]
        rebuild: bool,
    },
    /// Rank sections by meaning, optionally filtered on frontmatter.
    Query {
        text: String,
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// `key=value`, `key!=value`, or bare `key` for presence. Repeatable.
        /// On a list value, `=` reads as membership. `!=` also passes when the
        /// key is absent, so a filter never silently drops unannotated files.
        #[arg(long = "where", value_name = "PRED")]
        wheres: Vec<String>,
        /// Drop a section whose identity appears in any indexed section's KEY.
        /// Repeatable. An anti-join: with `--exclude-pointed-by supersedes`, a
        /// record another record supersedes stops being a candidate. The values
        /// are collected from the whole index, not from what the other filters
        /// leave, because a superseded record is superseded whether or not the
        /// record that replaced it also answers this query.
        #[arg(long, value_name = "KEY")]
        exclude_pointed_by: Vec<String>,
        /// The frontmatter key holding a section's own identity, read by
        /// `--exclude-pointed-by`. No key is built in; this is the default only
        /// because most schemas spell it this way.
        #[arg(long, value_name = "KEY", default_value = "id")]
        identity: String,
        #[arg(long, short, default_value_t = 5)]
        limit: usize,
    },
    /// Report what the index covers.
    Status {
        #[arg(default_value = ".")]
        root: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Index {
            root,
            endpoint,
            model,
            max_chars,
            rebuild,
        } => cmd_index(&root, &endpoint, &model, max_chars, rebuild),
        Cmd::Query {
            text,
            root,
            wheres,
            exclude_pointed_by,
            identity,
            limit,
        } => cmd_query(&root, &text, &wheres, &exclude_pointed_by, &identity, limit),
        Cmd::Status { root } => cmd_status(&root),
    }
}

// ---------------------------------------------------------------- persistence

/// The pair a prefilter compares, alongside the hash it stands in for. Zero
/// for a clock the platform will not answer for, which makes the file look
/// changed and sends it to be read.
fn stamp(meta: &fs::Metadata, hash: u64) -> Stamp {
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos() as i64);
    Stamp { hash, len: meta.len(), mtime }
}

/// Open the index for reading and check the matrix against it. Nothing is
/// deserialised here: a caller asks the store for the part of a record it
/// actually decides on.
fn open_index(root: &Path) -> Result<Option<(Store, Meta)>> {
    if !store::db_path(root).exists() {
        return Ok(None);
    }
    let st = Store::open(root)?;
    if st.compacting()? {
        bail!(
            "a compaction of {} did not finish, so the records may name the wrong rows — rerun `folio index --rebuild`",
            store::vectors_path(root).display()
        );
    }
    let meta = st.meta()?;
    if meta.dim == 0 {
        return Ok(None);
    }
    let vecs = store::vectors_path(root);
    let bytes = fs::metadata(&vecs)?.len() as usize;
    let last = st.high_water()?;
    if bytes % (meta.dim * 4) != 0 || bytes / (meta.dim * 4) < last {
        bail!(
            "index is inconsistent: {} holds {bytes} bytes, which is not {last} whole rows at dim {} — rerun with --rebuild",
            vecs.display(),
            meta.dim
        );
    }
    Ok(Some((st, meta)))
}

/// The matrix as floats, mapped rather than read. Reading 367 MB costs 76 ms,
/// converting it to `f32` another 76 ms and splitting it per row 80 ms; mapping
/// it costs nothing and a scan over the map pages in only what it touches.
fn map_vectors(root: &Path) -> Result<Option<memmap2::Mmap>> {
    let vecs = store::vectors_path(root);
    if !vecs.exists() {
        return Ok(None);
    }
    let file = fs::File::open(&vecs)?;
    // Safety: the index is folio's own file. A concurrent `folio index`
    // rewrites it, and the consistency check above is what catches that.
    Ok(Some(unsafe { memmap2::Mmap::map(&file)? }))
}

/// A page-aligned map divides into `f32` exactly; anything else is not a
/// matrix folio wrote.
fn as_floats(map: &memmap2::Mmap) -> Result<&[f32]> {
    let (head, mid, tail) = unsafe { map.align_to::<f32>() };
    if !head.is_empty() || !tail.is_empty() {
        bail!("the vector file is not a whole number of aligned f32 values");
    }
    Ok(mid)
}

// --------------------------------------------------------------------- embed

#[derive(Serialize)]
struct EmbedReq<'a> {
    model: &'a str,
    input: &'a [String],
}

#[derive(Deserialize)]
struct EmbedResp {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    index: usize,
    embedding: Vec<f32>,
}

/// Vectors come back L2-normalized, so ranking is a plain dot product.
fn embed(endpoint: &str, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); inputs.len()];
    for (bi, batch) in inputs.chunks(BATCH).enumerate() {
        // A server that answered is not a server that is absent, and only one
        // of those is worth retrying. Status is handled here rather than raised
        // as a transport error so that the server's own sentence survives: it
        // is the sentence that says which input was too long for its batch.
        let mut response = ureq::post(endpoint)
            .config()
            .http_status_as_error(false)
            .build()
            .send_json(EmbedReq { model, input: batch })
            .with_context(|| format!("no answer from the endpoint at {endpoint}"))?;
        let status = response.status();
        if !status.is_success() {
            let said = response
                .body_mut()
                .read_to_string()
                .unwrap_or_else(|e| format!("(its body was unreadable: {e})"));
            bail!(
                "the endpoint at {endpoint} answered {status} for inputs {}..{}: {}",
                bi * BATCH,
                bi * BATCH + batch.len() - 1,
                said.trim()
            );
        }
        let resp: EmbedResp = response
            .body_mut()
            .read_json()
            .context("embedding response did not match the OpenAI schema")?;
        for item in resp.data {
            let at = bi * BATCH + item.index;
            let Some(slot) = out.get_mut(at) else {
                bail!("embedding response carried index {at}, past the {} sent", inputs.len());
            };
            *slot = normalize(item.embedding);
        }
    }
    if let Some(i) = out.iter().position(Vec::is_empty) {
        bail!("embedding response was missing input {i}");
    }
    Ok(out)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// -------------------------------------------------------------------- filters

enum Pred {
    Eq(String, String),
    Ne(String, String),
    Has(String),
}

fn parse_preds(raw: &[String]) -> Vec<Pred> {
    raw.iter()
        .map(|s| {
            if let Some((k, v)) = s.split_once("!=") {
                Pred::Ne(k.trim().to_string(), v.trim().to_string())
            } else if let Some((k, v)) = s.split_once('=') {
                Pred::Eq(k.trim().to_string(), v.trim().to_string())
            } else {
                Pred::Has(s.trim().to_string())
            }
        })
        .collect()
}

fn keeps(fm: &Map<String, Value>, preds: &[Pred]) -> bool {
    preds.iter().all(|p| match p {
        Pred::Has(k) => fm.contains_key(k),
        Pred::Eq(k, v) => fm.get(k).is_some_and(|got| holds(got, v)),
        Pred::Ne(k, v) => !fm.get(k).is_some_and(|got| holds(got, v)),
    })
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
fn scalar_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string().trim_matches('"').to_string(),
    }
}

/// Every value any section carries under one of `keys`, flattened out of lists.
/// This is the right side of the anti-join, and it is built from every row in
/// the index rather than from the rows a filter left.
fn pointed_at(rows: &[(usize, Map<String, Value>)], keys: &[String]) -> BTreeSet<String> {
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

// ------------------------------------------------------------------- commands

fn hash(bytes: &[u8]) -> u64 {
    let mut h = DefaultHasher::new();
    h.write(bytes);
    h.finish()
}

/// What a re-index did, for whoever asked for it to say so.
struct Indexed {
    files: usize,
    sections: usize,
    truncated: usize,
    dim: usize,
    reembedded: usize,
    retired: usize,
    dead: usize,
    compacted: usize,
}

/// Bring the index under `root` up to date with the files under it.
///
/// `Ok(None)` means another process holds the write lock and this one declined
/// to wait: a re-index nobody asked for is not worth blocking on, and one the
/// user asked for says so rather than hanging.
fn reindex(
    root: &Path,
    endpoint: &str,
    model: &str,
    max_chars: usize,
    rebuild: bool,
    wait: std::time::Duration,
) -> Result<Option<Indexed>> {
    let root = root
        .canonicalize()
        .with_context(|| format!("{} not found", root.display()))?;
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
    // A vector space belongs to one model at one endpoint; mixing them would
    // rank incomparable numbers against each other.
    let reuse = !rebuild && prev.dim > 0 && prev.model == model && prev.endpoint == endpoint;
    let prev_files = if reuse { st.files()? } else { HashMap::new() };
    if !reuse {
        st.clear()?;
        fs::write(store::vectors_path(&root), [])?;
    }
    // The row the first new vector takes. Read before anything is retired, so
    // that a crash during the write lands past every row a live record names.
    let base = st.high_water()?;

    // The walk is threaded because it dominated what was left of a re-index:
    // on 14,616 files it ran 449-804 ms in one thread and 172 ms across ten.
    // Results land in whatever order the threads finish, which is fine because
    // they are collected into maps and a set.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let found: Mutex<Vec<(String, Stamp, bool)>> = Mutex::new(Vec::new());
    let failed: Mutex<Vec<String>> = Mutex::new(Vec::new());

    ignore::WalkBuilder::new(&root)
        .threads(threads)
        .build_parallel()
        .run(|| {
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
    retired_paths.extend(
        prev_files
            .keys()
            .filter(|p| !current.contains_key(*p))
            .cloned(),
    );

    let mut fresh: Vec<Section> = Vec::new();
    for rel in &changed {
        let source = fs::read_to_string(root.join(rel))?;
        for mut s in sections::split(rel, &source) {
            if s.text.chars().count() > max_chars {
                s.text = s.text.chars().take(max_chars).collect();
                s.truncated = true;
            }
            fresh.push(s);
        }
    }

    let mut dim = if reuse { prev.dim } else { 0 };
    let mut placed: Vec<(usize, Section)> = Vec::new();
    if !fresh.is_empty() {
        let inputs: Vec<String> = fresh.iter().map(|s| s.text.clone()).collect();
        let vectors = embed(endpoint, model, &inputs)?;
        if dim == 0 {
            dim = vectors[0].len();
        }
        if let Some((i, v)) = vectors.iter().enumerate().find(|(_, v)| v.len() != dim) {
            bail!(
                "{} came back at dim {} while the index is dim {dim}",
                fresh[i].path,
                v.len()
            );
        }
        append(&root, base, dim, &vectors)?;
        placed = (base..).zip(fresh).collect();
    }

    let retired = st.apply(
        &Meta {
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            dim,
        },
        &current,
        &retired_paths,
        &placed,
    )?;

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
        dim,
        reembedded: changed.len(),
        retired,
        dead,
        compacted,
    }))
}

fn cmd_index(
    root: &Path,
    endpoint: &str,
    model: &str,
    max_chars: usize,
    rebuild: bool,
) -> Result<()> {
    // An index the user asked for waits a little for one already running, and
    // then says who it is waiting for rather than hanging on it.
    let wait = std::time::Duration::from_secs(10);
    let Some(r) = reindex(root, endpoint, model, max_chars, rebuild, wait)? else {
        bail!(
            "another folio is writing the index under {} — try again once it is done",
            root.display()
        );
    };
    println!(
        "indexed {} files · {} sections · dim {}",
        r.files, r.sections, r.dim
    );
    println!(
        "  {} re-embedded, {} replaced or removed, {} dead row(s)",
        r.reembedded, r.retired, r.dead
    );
    if r.truncated > 0 {
        println!(
            "  {} sections truncated at --max-chars {max_chars}",
            r.truncated
        );
    }
    if r.compacted > 0 {
        println!("  compacted: {} dead row(s) reclaimed", r.compacted);
    }
    Ok(())
}

/// Rows the matrix holds, live and dead.
fn rows_in_matrix(root: &Path, dim: usize) -> Result<usize> {
    if dim == 0 {
        return Ok(0);
    }
    let vecs = store::vectors_path(root);
    Ok(if vecs.exists() {
        fs::metadata(&vecs)?.len() as usize / (dim * 4)
    } else {
        0
    })
}

/// Write `vectors` at row `base`, leaving every row before it untouched. The
/// file is not truncated: rows past the end of the write are dead, and a write
/// that fails must not shorten the matrix under the records that name them.
fn append(root: &Path, base: usize, dim: usize, vectors: &[Vec<f32>]) -> Result<()> {
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

fn cmd_query(
    root: &Path,
    text: &str,
    wheres: &[String],
    exclude_pointed_by: &[String],
    identity: &str,
    limit: usize,
) -> Result<()> {
    let root = root.canonicalize()?;
    let Some((st, meta)) = open_index(&root)? else {
        bail!("no index under {} — run `folio index` first", root.display());
    };
    let map = map_vectors(&root)?.context("the index has records but no vectors")?;
    let floats = as_floats(&map)?;

    let q = embed(&meta.endpoint, &meta.model, &[text.to_string()])?
        .pop()
        .expect("one input yields one vector");
    let score = |slot: usize| dot(&q, &floats[slot * meta.dim..(slot + 1) * meta.dim]);

    // Frontmatter is read only when something decides on it. Without a filter
    // and without a join, ranking needs the vectors and the list of rows that
    // are still live, and nothing else: on 119,359 sections that is 10 ms of
    // slots against 180 ms of whole records.
    let preds = parse_preds(wheres);
    let (mut hits, dropped): (Vec<(f32, usize)>, usize) =
        if preds.is_empty() && exclude_pointed_by.is_empty() {
            let slots = st.slots()?;
            if slots.is_empty() {
                bail!("no index under {} — run `folio index` first", root.display());
            }
            (slots.into_iter().map(|slot| (score(slot), slot)).collect(), 0)
        } else {
            let rows = st.slots_with_fm()?;
            if rows.is_empty() {
                bail!("no index under {} — run `folio index` first", root.display());
            }
            // The right side of the anti-join is still built from every row in
            // the index, which is what `--exclude-pointed-by` means.
            let pointed = pointed_at(&rows, exclude_pointed_by);
            let superseded = |fm: &Map<String, Value>| {
                !pointed.is_empty()
                    && fm
                        .get(identity)
                        .is_some_and(|v| pointed.contains(&scalar_text(v)))
            };
            let kept: Vec<&(usize, Map<String, Value>)> =
                rows.iter().filter(|(_, fm)| keeps(fm, &preds)).collect();
            let dropped = kept.iter().filter(|(_, fm)| superseded(fm)).count();
            (
                kept.iter()
                    .filter(|(_, fm)| !superseded(fm))
                    .map(|(slot, _)| (score(*slot), *slot))
                    .collect(),
                dropped,
            )
        };
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Reported before the empty case, so that "nothing matched" is never the
    // only thing a caller hears when the anti-join is what emptied the result.
    if dropped > 0 {
        println!(
            "({dropped} section(s) dropped as pointed at by {})",
            exclude_pointed_by.join(", ")
        );
    }
    if hits.is_empty() {
        println!("no section passed the filter");
        return Ok(());
    }

    // Only the rows that won are read whole. The heading trail and the line
    // range are printed, never ranked on.
    let top: Vec<usize> = hits.iter().take(limit).map(|(_, slot)| *slot).collect();
    for (i, (s, (score, _))) in st.hydrate(&top)?.iter().zip(&hits).enumerate() {
        println!("#{}  {score:.3}  {}:{}-{}", i + 1, s.path, s.start, s.end);
        let mut trail = s.breadcrumb.clone();
        trail.push(s.heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
        println!("        {}", trail.join(" > "));
    }
    Ok(())
}

fn cmd_status(root: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let Some((st, meta)) = open_index(&root)? else {
        println!("no index under {}", root.display());
        return Ok(());
    };
    let counts = st.counts()?;
    if counts.sections == 0 {
        println!("no index under {}", root.display());
        return Ok(());
    }
    // The one place that reads every record whole: reporting which frontmatter
    // keys the corpus carries is a question about all of them.
    let rows = st.slots_with_fm()?;
    let keys: HashSet<&String> = rows.iter().flat_map(|(_, fm)| fm.keys()).collect();
    let mut keys: Vec<&&String> = keys.iter().collect();
    keys.sort();

    println!("{}", root.display());
    println!("  files       {}", counts.files);
    println!("  sections    {}", counts.sections);
    println!("  truncated   {}", counts.truncated);
    println!("  model       {} @ {}", meta.model, meta.endpoint);
    println!("  dim         {}", meta.dim);
    println!(
        "  frontmatter {}",
        if keys.is_empty() {
            "(none)".to_string()
        } else {
            keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", ")
        }
    );
    Ok(())
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
}

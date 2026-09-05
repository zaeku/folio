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
use serde_json::Value;
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

type Row = (Section, Vec<f32>);

/// Metadata and section records, without the matrix. A query maps the matrix
/// instead of reading it, so the two are loaded apart.
fn load_sections(root: &Path) -> Result<(Meta, Vec<Section>)> {
    if !store::db_path(root).exists() {
        return Ok((Meta::default(), Vec::new()));
    }
    let st = Store::open(root)?;
    let meta = st.meta()?;
    if meta.dim == 0 {
        return Ok((meta, Vec::new()));
    }
    let secs = st.sections()?;

    let vecs = store::vectors_path(root);
    let bytes = fs::metadata(&vecs)?.len() as usize;
    if bytes != secs.len() * meta.dim * 4 {
        bail!(
            "index is inconsistent: {} sections at dim {} need {} bytes and {} holds {bytes} — rerun with --rebuild",
            secs.len(),
            meta.dim,
            secs.len() * meta.dim * 4,
            vecs.display()
        );
    }
    Ok((meta, secs))
}

/// The whole index owned, which `index` needs because it rewrites it.
fn load(root: &Path) -> Result<(Meta, Vec<Row>)> {
    let (meta, secs) = load_sections(root)?;
    if secs.is_empty() {
        return Ok((meta, Vec::new()));
    }
    let dim = meta.dim;
    let raw = fs::read(store::vectors_path(root))?;
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok((
        meta,
        secs.into_iter()
            .zip(floats.chunks_exact(dim))
            .map(|(s, v)| (s, v.to_vec()))
            .collect(),
    ))
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

/// The vectors go down first and the records commit after. A crash in between
/// leaves rows at the end of the matrix that no record names, which the next
/// write discards; the other order would leave records naming rows that were
/// never written.
fn save(
    root: &Path,
    meta: &Meta,
    files: &HashMap<String, Stamp>,
    rows: &[Row],
) -> Result<()> {
    let mut st = Store::open(root)?;

    let mut w = BufWriter::new(fs::File::create(store::vectors_path(root))?);
    for (_, v) in rows {
        for x in v {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()?;

    let secs: Vec<Section> = rows.iter().map(|(s, _)| s.clone()).collect();
    st.replace(meta, files, &secs)
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

fn keeps(s: &Section, preds: &[Pred]) -> bool {
    preds.iter().all(|p| match p {
        Pred::Has(k) => s.fm.contains_key(k),
        Pred::Eq(k, v) => s.fm.get(k).is_some_and(|got| holds(got, v)),
        Pred::Ne(k, v) => !s.fm.get(k).is_some_and(|got| holds(got, v)),
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
fn pointed_at(rows: &[Section], keys: &[String]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for s in rows {
        for key in keys {
            match s.fm.get(key) {
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

fn cmd_index(
    root: &Path,
    endpoint: &str,
    model: &str,
    max_chars: usize,
    rebuild: bool,
) -> Result<()> {
    let root = root
        .canonicalize()
        .with_context(|| format!("{} not found", root.display()))?;

    let (prev, mut rows) = if rebuild {
        (Meta::default(), Vec::new())
    } else {
        load(&root)?
    };
    let prev_files = if rebuild {
        HashMap::new()
    } else {
        Store::open(&root)?.files()?
    };
    // A vector space belongs to one model at one endpoint; mixing them would
    // rank incomparable numbers against each other.
    let reuse = !rebuild
        && prev.dim > 0
        && prev.model == model
        && prev.endpoint == endpoint;
    if !reuse {
        rows.clear();
    }

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

    let before = rows.len();
    rows.retain(|(s, _)| current.contains_key(&s.path) && !changed.contains(&s.path));
    let dropped = before - rows.len();

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

    if !fresh.is_empty() {
        let inputs: Vec<String> = fresh.iter().map(|s| s.text.clone()).collect();
        let vectors = embed(endpoint, model, &inputs)?;
        rows.extend(fresh.into_iter().zip(vectors));
    }

    let dim = rows.first().map_or(0, |(_, v)| v.len());
    if let Some((s, v)) = rows.iter().find(|(_, v)| v.len() != dim) {
        bail!(
            "{} came back at dim {} while the index is dim {dim}",
            s.path,
            v.len()
        );
    }

    let truncated = rows.iter().filter(|(s, _)| s.truncated).count();
    save(
        &root,
        &Meta {
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            dim,
        },
        &current,
        &rows,
    )?;

    println!(
        "indexed {} files · {} sections · dim {dim}",
        rows
            .iter()
            .map(|(s, _)| &s.path)
            .collect::<HashSet<_>>()
            .len(),
        rows.len()
    );
    println!("  {} re-embedded, {dropped} replaced or removed", changed.len());
    if truncated > 0 {
        println!("  {truncated} sections truncated at --max-chars {max_chars}");
    }
    Ok(())
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
    let (meta, rows) = load_sections(&root)?;
    if rows.is_empty() {
        bail!("no index under {} — run `folio index` first", root.display());
    }
    let map = map_vectors(&root)?.context("the index has records but no vectors")?;
    let floats = as_floats(&map)?;

    let q = embed(&meta.endpoint, &meta.model, &[text.to_string()])?
        .pop()
        .expect("one input yields one vector");

    let preds = parse_preds(wheres);
    let pointed = pointed_at(&rows, exclude_pointed_by);
    let superseded = |s: &Section| {
        !pointed.is_empty()
            && s.fm
                .get(identity)
                .is_some_and(|v| pointed.contains(&scalar_text(v)))
    };

    let mut hits: Vec<(f32, &Section)> = rows
        .iter()
        .enumerate()
        .filter(|(_, s)| keeps(s, &preds) && !superseded(s))
        .map(|(i, s)| (dot(&q, &floats[i * meta.dim..(i + 1) * meta.dim]), s))
        .collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));

    // Reported before the empty case, so that "nothing matched" is never the
    // only thing a caller hears when the anti-join is what emptied the result.
    let dropped = if pointed.is_empty() {
        0
    } else {
        rows.iter()
            .filter(|s| keeps(s, &preds) && superseded(s))
            .count()
    };
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
    for (i, (score, s)) in hits.iter().take(limit).enumerate() {
        println!("#{}  {score:.3}  {}:{}-{}", i + 1, s.path, s.start, s.end);
        let mut trail = s.breadcrumb.clone();
        trail.push(s.heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
        println!("        {}", trail.join(" > "));
    }
    Ok(())
}

fn cmd_status(root: &Path) -> Result<()> {
    let root = root.canonicalize()?;
    let (meta, rows) = load_sections(&root)?;
    if rows.is_empty() {
        println!("no index under {}", root.display());
        return Ok(());
    }
    let keys: HashSet<&String> = rows.iter().flat_map(|s| s.fm.keys()).collect();
    let mut keys: Vec<&&String> = keys.iter().collect();
    keys.sort();

    println!("{}", root.display());
    println!(
        "  files       {}",
        rows.iter().map(|s| &s.path).collect::<HashSet<_>>().len()
    );
    println!("  sections    {}", rows.len());
    println!("  truncated   {}", rows.iter().filter(|s| s.truncated).count());
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

    fn row(fm: Value) -> Section {
        let section = Section {
            path: "d.md".into(),
            start: 1,
            end: 1,
            heading: None,
            breadcrumb: Vec::new(),
            fm: fm.as_object().expect("an object").clone(),
            truncated: false,
            text: String::new(),
        };
        section
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
        let identity = rows[0].fm.get("id").expect("an id");
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

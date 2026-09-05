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

/// The default character budget per section. Set below the 8192-token context
/// of the models folio is measured on rather than at a round number.
const MAX_CHARS: usize = 8_000;

/// Where folio looks for an endpoint when nothing else names one.
const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080/v1/embeddings";

/// The model name folio sends. Most single-model servers ignore it.
const DEFAULT_MODEL: &str = "default";

/// What `folio index` needs before an index exists to remember it.
///
/// `folio query` never reads this. An index records the endpoint and model it
/// was built with, and a vector space belongs to one of each, so the recorded
/// pair is the only correct answer for a corpus that has one.
#[derive(Default, Deserialize, Serialize)]
struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
}

/// The per-corpus config, meant to be committed with the corpus.
///
/// It sits beside the corpus rather than inside `.folio/`, because `.folio/` is
/// a derived index that belongs to whoever built it. Which model a corpus needs
/// is not derived: a Korean corpus and an English one can want different ones,
/// and everyone who indexes that corpus wants the same answer.
const PROJECT_CONFIG: &str = "folio.yaml";

fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config")
        });
    base.join("folio").join("config.yaml")
}

/// Read the config file, or fail.
///
/// A file that does not parse is not the same as no file. Falling back to the
/// default endpoint would index a corpus against a model the user did not
/// choose, and vectors from the wrong model are not detectable from a ranking.
fn read_config_at(path: &Path) -> Result<Config> {
    match fs::read_to_string(path) {
        Ok(text) => serde_yaml_ng::from_str(&text)
            .with_context(|| format!("{} is not valid YAML", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// Resolve one setting, and say where the answer came from.
///
/// A flag beats the environment, the environment beats the file, and the file
/// beats the built-in default. `folio config` prints the source because a
/// surprising endpoint is usually an environment variable someone forgot.
fn resolve(
    flag: Option<&str>,
    env_key: &str,
    project: Option<&str>,
    user: Option<&str>,
    default: &str,
) -> (String, &'static str) {
    if let Some(v) = flag {
        return (v.to_string(), "flag");
    }
    if let Some(v) = std::env::var(env_key).ok().filter(|v| !v.is_empty()) {
        return (v, "environment");
    }
    if let Some(v) = project {
        return (v.to_string(), PROJECT_CONFIG);
    }
    if let Some(v) = user {
        return (v.to_string(), "user config");
    }
    (default.to_string(), "default")
}

/// The budget to re-index with, given what the index recorded.
///
/// An index written before folio recorded the budget has none, which reads back
/// as zero. Zero is not a budget: taken literally it truncates every section to
/// the empty string and embeds that, which is a silent and total corruption of
/// the index rather than an error. So it means "not recorded" and nothing else,
/// and `--max-chars` will not accept it.
fn budget(recorded: usize) -> usize {
    if recorded == 0 { MAX_CHARS } else { recorded }
}

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
        ///
        /// Falls back to `FOLIO_ENDPOINT`, then to the config file, then to
        /// http://127.0.0.1:8080/v1/embeddings. `folio config` prints which one
        /// answered.
        #[arg(long)]
        endpoint: Option<String>,
        /// The model name to send. Falls back to `FOLIO_MODEL`, then to the
        /// config file, then to `default`.
        #[arg(long)]
        model: Option<String>,
        /// Cap on the text sent per section. Exceeding sections are marked.
        ///
        /// A character budget standing in for the model's token limit, which
        /// folio cannot see. It cannot bound a token count in general — a
        /// byte-level tokenizer can spend more than one token on a multi-byte
        /// character — so this default is set below the 8192-token context of
        /// the models folio is measured on rather than at a round number. Raise
        /// it for a longer-context model, and read the truncated count.
        #[arg(long, default_value_t = MAX_CHARS, value_parser = at_least_one)]
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
        /// Answer from the index as it stands, without re-indexing first.
        ///
        /// By default a query stats the files behind the rows it is about to
        /// return, and re-indexes if any of them has moved. That is the case
        /// where a stale index does harm rather than waste: the line range it
        /// names has shifted, so the caller reads the wrong lines. It costs one
        /// stat per returned row and nothing else when nothing has changed.
        ///
        /// The stat happens either way. A row whose file has moved is marked
        /// `(stale)` whether folio was allowed to fix it or not.
        #[arg(long)]
        no_refresh: bool,
    },
    /// Show what `folio index` would use, or write it to a config file.
    Config {
        #[command(subcommand)]
        action: Option<ConfigCmd>,
        /// The corpus whose `folio.yaml` is read or written.
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
    /// Ask the endpoint three questions folio's ranking depends on.
    Doctor {
        /// The corpus whose `folio.yaml` names the endpoint to test.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long)]
        endpoint: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// The section budget to test the endpoint's batch size against.
        #[arg(long, default_value_t = MAX_CHARS, value_parser = at_least_one)]
        max_chars: usize,
    },
    /// Print a service file that runs the embeddings server. Installs nothing.
    Unit {
        /// The corpus whose `folio.yaml` names the endpoint to serve.
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Write a launchd agent. The default on macOS.
        #[arg(long, conflicts_with = "systemd")]
        launchd: bool,
        /// Write a systemd user service. The default elsewhere.
        #[arg(long)]
        systemd: bool,
        /// The model repository the server should pull.
        #[arg(long, default_value = "keisuke-miyako/gte-modernbert-base-gguf")]
        hf: String,
        /// The file within that repository.
        #[arg(long, default_value = "gte-modernbert-base-Q8_0.gguf")]
        hf_file: String,
        /// The pooling this model wants. `cls` for gte-modernbert, `last` for
        /// the Qwen3-Embedding family. The server's own default is wrong for
        /// both, and a wrong one ranks badly without failing.
        #[arg(long, default_value = "cls")]
        pooling: String,
        /// Context, and the physical batch, in tokens. Must exceed your longest
        /// section: an encoder needs its whole input in one batch.
        #[arg(long, default_value_t = 8192)]
        context: usize,
    },
    /// Report what the index covers.
    Status {
        #[arg(default_value = ".")]
        root: PathBuf,
    },
}

#[derive(Subcommand)]
enum ConfigCmd {
    /// Write one setting to a config file, creating it if it is absent.
    Set {
        /// `endpoint` or `model`.
        key: String,
        value: String,
        /// Write `folio.yaml` beside the corpus instead of the user's config.
        /// Commit that file, and everyone who indexes the corpus embeds it the
        /// same way.
        #[arg(long)]
        project: bool,
    },
}

fn at_least_one(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("a section budget of 0 characters would embed nothing".into()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Index {
            root,
            endpoint,
            model,
            max_chars,
            rebuild,
        } => cmd_index(
            &root,
            endpoint.as_deref(),
            model.as_deref(),
            max_chars,
            rebuild,
        ),
        Cmd::Query {
            text,
            root,
            wheres,
            exclude_pointed_by,
            identity,
            limit,
            no_refresh,
        } => cmd_query(
            &root,
            &text,
            &wheres,
            &exclude_pointed_by,
            &identity,
            limit,
            !no_refresh,
        ),
        Cmd::Config { action, root } => cmd_config(action, &root),
        Cmd::Doctor { root, endpoint, model, max_chars } => cmd_doctor(
            &root,
            endpoint.as_deref(),
            model.as_deref(),
            max_chars,
        ),
        Cmd::Unit { root, launchd, systemd, hf, hf_file, pooling, context } => cmd_unit(
            &root, launchd, systemd, &hf, &hf_file, &pooling, context,
        ),
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
            // The first line is unchanged on purpose: a fence in the decision
            // layer reads "endpoint at" to tell an outage from a violation.
            .with_context(|| {
                format!(
                    "no answer from the endpoint at {endpoint}\n\n\
                     folio runs no model of its own. Start an OpenAI-compatible \
                     /v1/embeddings service, then name it:\n\n    \
                     folio config set endpoint <url>\n\n\
                     One measured server, and the two flags it needs, are in \
                     https://github.com/zaeku/folio#install"
                )
            })?;
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
    // A vector space belongs to one model at one endpoint; mixing them would
    // rank incomparable numbers against each other.
    let reuse = !rebuild && prev.dim > 0 && prev.model == model && prev.endpoint == endpoint;
    if !reuse && !rebuild && prev.dim > 0 {
        // Discarding is the decision working. Saying so is the difference
        // between that and a re-index someone cannot account for, and the
        // usual cause is an exported variable from another corpus.
        println!(
            "the index was built by {} at {}, and this run uses {model} at {endpoint} — \
             re-embedding every section",
            prev.model, prev.endpoint
        );
    }
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
            max_chars,
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
    endpoint: Option<&str>,
    model: Option<&str>,
    max_chars: usize,
    rebuild: bool,
) -> Result<()> {
    let proj = read_config_at(&root.join(PROJECT_CONFIG))?;
    let user = read_config_at(&config_path())?;
    let (endpoint, _) = resolve(
        endpoint,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (model, _) = resolve(
        model,
        "FOLIO_MODEL",
        proj.model.as_deref(),
        user.model.as_deref(),
        DEFAULT_MODEL,
    );
    let (endpoint, model) = (endpoint.as_str(), model.as_str());
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

/// The rows a query keeps, best first, and how many the anti-join dropped.
fn rank(
    st: &Store,
    meta: &store::Meta,
    floats: &[f32],
    q: &[f32],
    preds: &[Pred],
    exclude_pointed_by: &[String],
    identity: &str,
) -> Result<(Vec<(f32, usize)>, usize)> {
    let score = |slot: usize| dot(q, &floats[slot * meta.dim..(slot + 1) * meta.dim]);

    // Frontmatter is read only when something decides on it. Without a filter
    // and without a join, ranking needs the vectors and the list of rows that
    // are still live, and nothing else: on 119,359 sections that is 10 ms of
    // slots against 180 ms of whole records.
    let (mut hits, dropped): (Vec<(f32, usize)>, usize) =
        if preds.is_empty() && exclude_pointed_by.is_empty() {
            let slots = st.slots()?;
            (slots.into_iter().map(|slot| (score(slot), slot)).collect(), 0)
        } else {
            let rows = st.slots_with_fm()?;
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
                rows.iter().filter(|(_, fm)| keeps(fm, preds)).collect();
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

fn cmd_query(
    root: &Path,
    text: &str,
    wheres: &[String],
    exclude_pointed_by: &[String],
    identity: &str,
    limit: usize,
    refresh: bool,
) -> Result<()> {
    let root = root.canonicalize()?;
    let preds = parse_preds(wheres);
    let mut q: Option<Vec<f32>> = None;
    // At most one refresh. A result still stale after re-indexing means the
    // files are moving while folio reads them, and saying so beats looping.
    let mut refreshed = false;

    loop {
        let Some((st, meta)) = open_index(&root)? else {
            bail!("no index under {} — run `folio index` first", root.display());
        };
        let map = map_vectors(&root)?.context("the index has records but no vectors")?;
        let floats = as_floats(&map)?;
        if q.is_none() {
            q = Some(
                embed(&meta.endpoint, &meta.model, &[text.to_string()])?
                    .pop()
                    .expect("one input yields one vector"),
            );
        }
        let q = q.as_deref().expect("embedded above");

        let (hits, dropped) = rank(
            &st,
            &meta,
            floats,
            q,
            &preds,
            exclude_pointed_by,
            identity,
        )?;
        let top: Vec<usize> = hits.iter().take(limit).map(|(_, slot)| *slot).collect();
        let rows = st.hydrate(&top)?;

        // Always asked, whatever `refresh` says: one stat per returned row is
        // free, and a caller told to answer from the index as it stands is the
        // one who most needs to know where it does not.
        let stale = moved_since_indexed(&st, &root, &rows)?;
        if refresh && !refreshed && !stale.is_empty() {
            drop(map);
            drop(st);
            // Declined rather than waited: another folio is already writing,
            // and its result will be at least as fresh as this one's would be.
            if let Some(r) = reindex(
                &root,
                &meta.endpoint,
                &meta.model,
                budget(meta.max_chars),
                false,
                std::time::Duration::ZERO,
            )? {
                println!(
                    "({} of the files behind this result had changed; {} file(s) re-embedded before answering)",
                    stale.len(),
                    r.reembedded
                );
                refreshed = true;
                continue;
            }
        }

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
        for (i, (s, (score, _))) in rows.iter().zip(&hits).enumerate() {
            // A row still stale here is one the refresh could not take, so the
            // line range may have moved. Said out loud rather than left to the
            // caller to discover by reading the wrong lines.
            let mark = if stale.contains(&s.path) { "  (stale)" } else { "" };
            println!("#{}  {score:.3}  {}:{}-{}{mark}", i + 1, s.path, s.start, s.end);
            let mut trail = s.breadcrumb.clone();
            trail.push(s.heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
            println!("        {}", trail.join(" > "));
        }
        return Ok(());
    }
}

fn cmd_config(action: Option<ConfigCmd>, root: &Path) -> Result<()> {
    let user_path = config_path();
    let proj_path = root.join(PROJECT_CONFIG);

    if let Some(ConfigCmd::Set { key, value, project }) = action {
        let path = if project { proj_path.clone() } else { user_path.clone() };
        let mut cfg = read_config_at(&path)?;
        match key.as_str() {
            "endpoint" => cfg.endpoint = Some(value),
            "model" => cfg.model = Some(value),
            other => bail!("no setting named {other} — folio config holds endpoint and model"),
        }
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            fs::create_dir_all(dir).with_context(|| format!("cannot create {}", dir.display()))?;
        }
        let text = serde_yaml_ng::to_string(&cfg)?;
        fs::write(&path, text).with_context(|| format!("cannot write {}", path.display()))?;
        println!("wrote {}", path.display());
    }

    let proj = read_config_at(&proj_path)?;
    let user = read_config_at(&user_path)?;
    let (endpoint, e_src) = resolve(
        None,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (model, m_src) = resolve(
        None,
        "FOLIO_MODEL",
        proj.model.as_deref(),
        user.model.as_deref(),
        DEFAULT_MODEL,
    );
    println!("  endpoint  {endpoint}  ({e_src})");
    println!("  model     {model}  ({m_src})");
    for (label, path) in [("corpus", &proj_path), ("user  ", &user_path)] {
        println!(
            "  {label}    {}{}",
            path.display(),
            if path.exists() { "" } else { "  (not written yet)" }
        );
    }
    // A query is answered from what the index recorded, so this pair reaches
    // `folio index` and nothing else.
    println!("\n`folio query` uses whatever the index recorded; run `folio status` for that.");
    Ok(())
}

/// The host and port a service should bind, read from the endpoint folio uses.
fn host_port(endpoint: &str) -> Result<(String, u16)> {
    let rest = endpoint.split_once("://").map_or(endpoint, |(_, r)| r);
    let authority = rest.split(['/', '?']).next().unwrap_or_default();
    let (host, port) = authority
        .rsplit_once(':')
        .with_context(|| format!("{endpoint} names no port, so a service cannot bind it"))?;
    let port: u16 = port
        .parse()
        .with_context(|| format!("{port} is not a port number"))?;
    Ok((host.to_string(), port))
}

/// Print a service file. Writing it and loading it stay with the reader.
///
/// folio speaks HTTP and nothing else, so it does not know which model file or
/// pooling mode the server needs. Those arrive as flags, defaulting to the pair
/// docs/measurements.md was measured on. What folio does know is the port its
/// own configuration points at, which is the part that is easy to get wrong.
fn cmd_unit(
    root: &Path,
    launchd: bool,
    systemd: bool,
    hf: &str,
    hf_file: &str,
    pooling: &str,
    context: usize,
) -> Result<()> {
    let proj = read_config_at(&root.join(PROJECT_CONFIG))?;
    let user = read_config_at(&config_path())?;
    let (endpoint, _) = resolve(
        None,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (host, port) = host_port(&endpoint)?;

    // launchd starts a job with a bare environment, so an unqualified name is
    // not found. The path is resolved here rather than left to the reader.
    let server = which_llama_server();
    let args = [
        "--embeddings".to_string(),
        "-hf".to_string(), hf.to_string(),
        "--hf-file".to_string(), hf_file.to_string(),
        "--pooling".to_string(), pooling.to_string(),
        "-c".to_string(), context.to_string(),
        "-b".to_string(), context.to_string(),
        "-ub".to_string(), context.to_string(),
        "--host".to_string(), host,
        "--port".to_string(), port.to_string(),
    ];

    let use_launchd = if launchd || systemd { launchd } else { cfg!(target_os = "macos") };
    if use_launchd {
        let argv: String = std::iter::once(server.clone())
            .chain(args)
            .map(|a| format!("    <string>{a}</string>\n"))
            .collect();
        print!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<!-- Serves {endpoint} for folio. It runs from login until you unload it;
     llama-server cannot be started on demand, because it binds its own socket
     rather than accepting one from launchd. It is not free while idle:
     docs/measurements.md in the folio repository carries what it costs. -->
<plist version="1.0">
<dict>
  <key>Label</key><string>dev.folio.embeddings</string>
  <key>ProgramArguments</key>
  <array>
{argv}  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/tmp/folio-embeddings.log</string>
  <key>StandardErrorPath</key><string>/tmp/folio-embeddings.log</string>
</dict>
</plist>
"#
        );
    } else {
        let argv = args.join(" ");
        print!(
            "# Serves {endpoint} for folio. It runs from login until you stop it;\n\
             # llama-server cannot be socket-activated, because it binds its own\n\
             # socket rather than accepting one from systemd. It is not free while\n\
             # idle: docs/measurements.md in the folio repository carries the cost.\n\
             [Unit]\n\
             Description=Embeddings endpoint for folio\n\
             After=network.target\n\
             \n\
             [Service]\n\
             ExecStart={server} {argv}\n\
             Restart=on-failure\n\
             \n\
             [Install]\n\
             WantedBy=default.target\n"
        );
    }
    Ok(())
}

/// The server's absolute path, or the bare name with a note when it is absent.
fn which_llama_server() -> String {
    let name = "llama-server";
    for dir in std::env::var("PATH").unwrap_or_default().split(':') {
        let candidate = Path::new(dir).join(name);
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    eprintln!("{name} is not on PATH; the service file names it unqualified, and a service manager will not find it");
    name.to_string()
}

/// The longest section under `root`, capped at the budget, for the batch test.
///
/// Synthetic filler would answer the wrong question. Repeated words tokenize
/// far more cheaply than prose, so a probe built from them fits a batch that
/// the corpus itself would overflow.
fn longest_section(root: &Path, max_chars: usize) -> (String, String) {
    let mut best = String::new();
    let mut where_from = String::new();
    for entry in ignore::WalkBuilder::new(root).build().flatten() {
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Ok(text) = fs::read_to_string(path) else {
            continue;
        };
        for sec in sections::split(&path.to_string_lossy(), &text) {
            if sec.text.chars().count() > best.chars().count() {
                where_from = format!("the longest section in {}", sec.path);
                best = sec.text;
            }
        }
    }
    if best.is_empty() {
        // ponytail: prose-like filler at roughly four characters per token,
        // which is English. A corpus is the better probe whenever one is there.
        let filler = "The quarterly figures arrived late again this morning. ";
        return (
            filler.repeat(max_chars / filler.len() + 1)[..max_chars].to_string(),
            "synthetic English prose, no corpus under this root".to_string(),
        );
    }
    if best.chars().count() > max_chars {
        best = best.chars().take(max_chars).collect();
        where_from = format!("{where_from}, cut to the budget");
    }
    (best, where_from)
}

/// Ask the endpoint what folio's ranking assumes of it.
///
/// Two of its failures are silent. A section longer than the server's physical
/// batch comes back as an HTTP error rather than a truncated vector, and a
/// pooling mode the model was not trained for returns vectors that rank badly
/// while looking like vectors. Neither shows in a result list.
fn cmd_doctor(
    root: &Path,
    endpoint: Option<&str>,
    model: Option<&str>,
    max_chars: usize,
) -> Result<()> {
    let proj = read_config_at(&root.join(PROJECT_CONFIG))?;
    let user = read_config_at(&config_path())?;
    let (endpoint, e_src) = resolve(
        endpoint,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (model, _) = resolve(
        model,
        "FOLIO_MODEL",
        proj.model.as_deref(),
        user.model.as_deref(),
        DEFAULT_MODEL,
    );
    println!("  endpoint   {endpoint}  ({e_src})");
    println!("  model      {model}");

    let mut failed = false;

    // 1. It answers, and with how many dimensions.
    let t = std::time::Instant::now();
    let probe = match embed(&endpoint, &model, &["a sentence to embed".to_string()]) {
        Ok(v) => v,
        Err(e) => {
            println!("  reachable  no");
            return Err(e);
        }
    };
    println!(
        "  reachable  yes, {} dimensions, {} ms for one input",
        probe[0].len(),
        t.elapsed().as_millis()
    );

    // 2. The longest thing this corpus would send fits the server's physical
    //    batch. `-b`/`-ub` below it is an HTTP 500, not a truncation. The text
    //    comes from the corpus because characters are not tokens: the same
    //    8,000 characters are about 2,000 tokens of English and several times
    //    that in a language that does not spell words with spaces.
    let (long, from) = longest_section(root, max_chars);
    match embed(&endpoint, &model, &[long.clone()]) {
        Ok(_) => println!("  long input {} characters accepted, {from}", long.chars().count()),
        Err(e) => {
            failed = true;
            println!(
                "  long input {} characters refused, {from} — raise the server's batch \
                 (-b and -ub) to at least your longest section, or lower --max-chars\n             {}",
                long.chars().count(),
                e.to_string().lines().next().unwrap_or_default()
            );
        }
    }

    // 3. Similarity still has structure. A pooling mode the model was not
    //    trained for either inverts this pair or collapses every distance.
    let probes = [
        "The cat sat on the warm windowsill.".to_string(),
        "A cat was sitting on the sunny window ledge.".to_string(),
        "Quarterly revenue is recognised when the goods ship.".to_string(),
    ];
    match embed(&endpoint, &model, &probes) {
        Err(e) => {
            failed = true;
            println!("  structure  could not measure: {e}");
        }
        Ok(v) => {
            let (near, far) = (dot(&v[0], &v[1]), dot(&v[0], &v[2]));
            print!("  structure  paraphrase {near:.3}, unrelated {far:.3}");
            // ponytail: two thresholds on one English triple. It catches an
            // inverted or collapsed space, not a merely mediocre one; a real
            // quality measure is the unmeasured question in docs/measurements.md.
            let inverted = near <= far;
            let collapsed = far > 0.95;
            if inverted {
                println!(" — an unrelated sentence ranks as close as a paraphrase");
            } else if collapsed {
                println!(" — unrelated sentences are nearly identical");
            } else {
                println!(" — ok");
            }
            if inverted || collapsed {
                failed = true;
                println!("             check the server's pooling: this model family wants one \
                          specific mode, and the default is wrong for some of them");
            }
        }
    }

    println!("\n  The probe is English. On a corpus in another language the last");
    println!("  question says less, and the first two say the same.");
    if failed {
        bail!("the endpoint is reachable but answers in a way folio's ranking cannot use");
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
    fn a_service_binds_what_the_endpoint_names() {
        assert_eq!(
            host_port("http://127.0.0.1:8080/v1/embeddings").unwrap(),
            ("127.0.0.1".to_string(), 8080)
        );
        assert_eq!(
            host_port("https://box.local:9999/v1/embeddings").unwrap(),
            ("box.local".to_string(), 9999)
        );
        // A port folio cannot read is a service that would bind the wrong one.
        assert!(host_port("http://example.com/v1/embeddings").is_err());
    }

        #[test]
    fn each_scope_beats_the_one_below_it() {
        const K: &str = "FOLIO_TEST_ENDPOINT";
        // SAFETY: single-threaded within this test, and the key is unique to it.
        unsafe { std::env::set_var(K, "from-env") };
        let (proj, user) = (Some("from-corpus"), Some("from-user"));

        let (v, src) = resolve(Some("from-flag"), K, proj, user, "from-default");
        assert_eq!((v.as_str(), src), ("from-flag", "flag"));

        let (v, src) = resolve(None, K, proj, user, "from-default");
        assert_eq!((v.as_str(), src), ("from-env", "environment"));

        unsafe { std::env::remove_var(K) };
        let (v, src) = resolve(None, K, proj, user, "from-default");
        assert_eq!((v.as_str(), src), ("from-corpus", PROJECT_CONFIG));

        let (v, src) = resolve(None, K, None, user, "from-default");
        assert_eq!((v.as_str(), src), ("from-user", "user config"));

        let (v, src) = resolve(None, K, None, None, "from-default");
        assert_eq!((v.as_str(), src), ("from-default", "default"));
    }

    #[test]
    fn an_unrecorded_budget_is_not_a_budget_of_zero() {
        // The case: an index written before folio recorded the budget reads
        // back as zero, and a refresh took it literally, truncated every
        // section of the file it was refreshing to nothing, and embedded the
        // empty string. Fifteen sections of MDN went silently blank that way.
        assert_eq!(budget(0), MAX_CHARS);
        assert_eq!(budget(1), 1);
        assert_eq!(budget(12_000), 12_000);
        assert!(at_least_one("0").is_err(), "and nobody can ask for it on purpose");
        assert_eq!(at_least_one("1"), Ok(1));
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

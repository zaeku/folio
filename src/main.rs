//! folio — section-scoped semantic search over a markdown corpus.
//!
//! The index holds references (path, line range, frontmatter), never bodies.
//! Retrieval names sections to read; reading the file is a separate step, so a
//! stale index costs a wasted candidate and never a wrong quotation.

mod sections;
mod store;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use sections::Section;
use serde::{Deserialize, Serialize};
use store::{Meta, Stamp, Store};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::hash::{DefaultHasher, Hasher};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::UNIX_EPOCH;

const BATCH: usize = 32;

/// The text whose embedding says which weights answered. Its value does not
/// matter; that every run sends the same one does. An index stores the text it
/// was built with, so changing this affects indexes built after it and no other.
const FINGERPRINT: &str = "folio probes the endpoint to learn which weights answered.";

/// Where the same weights sit. Measured 2026-09-06 on gte-modernbert-base-Q8_0
/// under llama.cpp: one input embedded twice in a request is bit-identical, and
/// at a different position of a 32-input batch it differs by 1.1e-7. Different
/// weights differ by 0.99. Nothing observed sits between.
const SAME_SPACE: f32 = 0.999;

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
        /// The corpus to rank. Repeatable.
        ///
        /// Several roots are ranked as one answer, and must be one vector
        /// space: the same weights, proven by the fingerprint each index
        /// carries, and the same character budget. A root with no index, or one
        /// folio cannot show belongs with the others, is refused rather than
        /// left out. They may cover the same files; a section two of them hold
        /// is returned once, and the query says how many it collapsed.
        #[arg(long, default_value = ".")]
        root: Vec<PathBuf>,
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
        /// Print one `path:start-end` per line and nothing else.
        ///
        /// For a caller that feeds the ranges to something that reads files.
        /// Notices and a stale row's warning go to stderr, so stdout stays one
        /// reference per line.
        #[arg(long)]
        paths_only: bool,
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
        /// Which server the unit runs. `llama.cpp` prints `llama-server`'s
        /// command, `tei` prints `text-embeddings-router`'s.
        ///
        /// The default is what folio has always printed, so an invocation
        /// written before this flag existed produces the same file.
        #[arg(long, value_enum, default_value_t = Backend::LlamaCpp)]
        backend: Backend,
        /// Write a launchd agent. The default on macOS.
        #[arg(long, conflicts_with = "systemd")]
        launchd: bool,
        /// Write a systemd user service. The default elsewhere.
        #[arg(long)]
        systemd: bool,
        /// The model repository the server should pull. Defaults to the GGUF
        /// conversion measured for `llama.cpp`, and to the safetensors the
        /// same model publishes for `tei`.
        #[arg(long)]
        hf: Option<String>,
        /// The file within that repository. `llama.cpp` only: `tei` reads
        /// safetensors and picks the file itself.
        #[arg(long)]
        hf_file: Option<String>,
        /// The pooling this model wants. `cls` for gte-modernbert, `last` for
        /// the Qwen3-Embedding family. The server's own default is wrong for
        /// both, and a wrong one ranks badly without failing.
        ///
        /// `llama.cpp` only: `tei` reads pooling from the model.
        #[arg(long)]
        pooling: Option<String>,
        /// Context, and the physical batch, in tokens. Must exceed your longest
        /// section: an encoder needs its whole input in one batch. Reaches
        /// `--max-batch-tokens` on `tei`, which needs it for the same reason.
        #[arg(long, default_value_t = 8192)]
        context: usize,
    },
    /// Report what the index covers.
    Status {
        /// The corpus to report on. Repeatable, one block each.
        #[arg(default_value = ".")]
        root: Vec<PathBuf>,
        /// List the sections that were cut to the budget, rather than counting
        /// them. Each was ranked on part of its text.
        #[arg(long)]
        truncated: bool,
    },
}

/// A server `folio unit` knows how to start.
///
/// folio runs no model and learns nothing about one from this; what it holds
/// per backend is a recipe, and D-01M1XRDKA9FJ1V says why holding one beats
/// handing the caller a command to write.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Backend {
    #[value(name = "llama.cpp")]
    LlamaCpp,
    #[value(name = "tei")]
    Tei,
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
    // A command that is piped into `head` has its stdout closed early. Rust
    // ignores SIGPIPE, so the next `println!` panics and prints a backtrace
    // notice over the output the reader was already reading. Restoring the
    // default ends this process the way every other command in a pipeline
    // ends.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };

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
            paths_only,
            no_refresh,
        } => cmd_query(
            &root,
            &text,
            &wheres,
            &exclude_pointed_by,
            &identity,
            limit,
            paths_only,
            !no_refresh,
        ),
        Cmd::Config { action, root } => cmd_config(action, &root),
        Cmd::Doctor { root, endpoint, model, max_chars } => cmd_doctor(
            &root,
            endpoint.as_deref(),
            model.as_deref(),
            max_chars,
        ),
        Cmd::Unit { root, backend, launchd, systemd, hf, hf_file, pooling, context } => cmd_unit(
            &root, backend, launchd, systemd, hf.as_deref(), hf_file.as_deref(),
            pooling.as_deref(), context,
        ),
        Cmd::Status { root, truncated } => cmd_status(&root, truncated),
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

/// The largest budget this endpoint accepts for this corpus's longest section.
///
/// folio budgets characters and the server counts tokens, and the ratio belongs
/// to the corpus rather than to folio: on the model in docs/measurements/,
/// English prose runs 3.66 characters per token and Korean 0.66. So a budget
/// that fits one corpus overflows another by a factor of five, and the only
/// authority on the difference is the server that has to accept the input.
///
/// Halving rather than searching: the answer is used to cut text, so landing
/// under the limit matters and landing exactly on it does not.
fn calibrate(endpoint: &str, model: &str, longest: &str, budget: usize) -> Result<usize> {
    /// Below this, a section is too short to carry a section's meaning, and a
    /// server refusing it is refusing for some other reason.
    const FLOOR: usize = 256;

    let mut budget = budget;
    loop {
        let probe: String = longest.chars().take(budget).collect();
        match embed(endpoint, model, &[probe]) {
            Ok(_) => return Ok(budget),
            // Nothing was measured about the input, so there is nothing to shrink.
            Err(EmbedError::NoAnswer(e)) => return Err(e),
            Err(EmbedError::Answered(e)) if budget <= FLOOR => return Err(e),
            Err(EmbedError::Answered(_)) => {
                budget /= 2;
                println!("  the endpoint refused the longest section; trying {budget} characters");
            }
        }
    }
}

/// Why an embedding call did not return vectors.
///
/// A server that answered and a server that is absent are different failures,
/// and only the first says anything about the input that was sent. Calibration
/// shrinks a budget on the first and stops on the second, so the difference is
/// carried in the type rather than recovered from the message.
enum EmbedError {
    NoAnswer(anyhow::Error),
    Answered(anyhow::Error),
}

impl From<EmbedError> for anyhow::Error {
    fn from(e: EmbedError) -> Self {
        match e {
            EmbedError::NoAnswer(e) | EmbedError::Answered(e) => e,
        }
    }
}

/// Vectors come back L2-normalized, so ranking is a plain dot product.
fn embed(endpoint: &str, model: &str, inputs: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
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
            })
            .map_err(EmbedError::NoAnswer)?;
        let status = response.status();
        if !status.is_success() {
            let said = response
                .body_mut()
                .read_to_string()
                .unwrap_or_else(|e| format!("(its body was unreadable: {e})"));
            return Err(EmbedError::Answered(anyhow::anyhow!(
                "the endpoint at {endpoint} answered {status} for inputs {}..{}: {}",
                bi * BATCH,
                bi * BATCH + batch.len() - 1,
                said.trim()
            )));
        }
        let resp: EmbedResp = response
            .body_mut()
            .read_json()
            .context("embedding response did not match the OpenAI schema")
            .map_err(EmbedError::Answered)?;
        for item in resp.data {
            let at = bi * BATCH + item.index;
            let Some(slot) = out.get_mut(at) else {
                return Err(EmbedError::Answered(anyhow::anyhow!(
                    "embedding response carried index {at}, past the {} sent",
                    inputs.len()
                )));
            };
            *slot = normalize(item.embedding);
        }
    }
    if let Some(i) = out.iter().position(Vec::is_empty) {
        return Err(EmbedError::Answered(anyhow::anyhow!(
            "embedding response was missing input {i}"
        )));
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

/// How close the endpoint answering now is to the one that built the index.
///
/// A different length is a different space and not a near miss, so it scores 0
/// rather than comparing the dimensions they happen to share.
fn same_space(now: &[f32], then: &[f32]) -> f32 {
    if now.len() != then.len() || then.is_empty() {
        return 0.0;
    }
    dot(now, then)
}

// -------------------------------------------------------------------- filters

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
struct Pred {
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
        anyhow!("--where {whole}: {why}. folio compares text, and the whole grammar is {WHERE_GRAMMAR}")
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
fn parse_preds(raw: &[String]) -> Result<(Vec<Pred>, Vec<String>)> {
    let mut preds = Vec::new();
    let mut hints = Vec::new();
    for whole in raw {
        let pieces: Vec<&str> = whole.split('|').collect();
        let any: Vec<Term> = pieces
            .iter()
            .map(|piece| parse_term(piece, whole))
            .collect::<Result<_>>()?;
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

fn keeps(fm: &Map<String, Value>, preds: &[Pred]) -> bool {
    preds
        .iter()
        .all(|p| p.any.iter().any(|t| satisfies(fm, t)))
}

/// Which `--where` kept nothing at all, for a filter that emptied the result.
///
/// "no section passed the filter" says that something did not match and not
/// what. Each predicate is applied alone, so the one that emptied the result is
/// named — including the case where a whole expression was read as one value.
fn blames(rows: &[(usize, Map<String, Value>)], preds: &[Pred]) -> Vec<String> {
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
    /// The budget these sections were cut at, which is `--max-chars` unless the
    /// endpoint refused it.
    budget: usize,
    dim: usize,
    reembedded: usize,
    moved: usize,
    retired: usize,
    dead: usize,
    compacted: usize,
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
        let Some(from) = gone
            .get_mut(&current[to].hash)
            .and_then(|paths| paths.pop())
        else {
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
    // A vector space is what the endpoint returns, not what it is called: a
    // server restarted on other weights keeps the URL and the model string it
    // had. So the index is kept or discarded on what answers now, and the probe
    // sent is the one this index was built with rather than today's default.
    let fingerprint_text = if prev.fingerprint.is_empty() { FINGERPRINT } else { prev.fingerprint.as_str() }.to_string();
    let mut fingerprint: Vec<f32> = Vec::new();
    let mut drifted: Option<f32> = None;
    if !rebuild && prev.dim > 0 {
        fingerprint = embed(endpoint, model, &[fingerprint_text.clone()])?
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
        let fitted = calibrate(endpoint, model, &longest, max_chars)?;
        if fitted < max_chars {
            println!("  budget for this run: {fitted} characters, not {max_chars}");
            max_chars = fitted;
            fresh = fresh
                .into_iter()
                .flat_map(|s| sections::to_budget(s, max_chars))
                .collect();
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
        let v = embed(endpoint, model, &[FINGERPRINT.to_string()])?
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
        let vectors = embed(endpoint, model, &inputs)?;
        if dim == 0 {
            dim = vectors[0].len();
        }
        if let Some((i, v)) = vectors.iter().enumerate().find(|(_, v)| v.len() != dim) {
            bail!(
                "{} came back at dim {} while the index is dim {dim}",
                batch[i].path,
                v.len()
            );
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
    if r.moved > 0 {
        println!("  {} moved, keeping the vectors they had", r.moved);
    }
    if r.truncated > 0 {
        println!(
            "  {} sections truncated at {} characters",
            r.truncated, r.budget
        );
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
        let c = r
            .canonicalize()
            .with_context(|| format!("{} cannot be read", r.display()))?;
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
    if out.len() > 1 {
        if let Some(o) = out.iter().find(|o| o.meta.fingerprint.is_empty()) {
            bail!(
                "the index under {} carries no fingerprint, so folio cannot show it holds the \
                 same vectors as the others — run `folio index --rebuild` there",
                o.root.display()
            );
        }
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
fn embed_and_prove(opened: &[Opened], text: &str) -> Result<Vec<f32>> {
    let first = &opened.first().expect("a query names at least one root").meta;
    let mut inputs: Vec<String> = Vec::new();
    for o in opened {
        if !o.meta.fingerprint.is_empty() && !inputs.contains(&o.meta.fingerprint) {
            inputs.push(o.meta.fingerprint.clone());
        }
    }
    inputs.push(text.to_string());
    let mut vectors = embed(&first.endpoint, &first.model, &inputs)?;
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
fn rank(
    opened: &[Opened],
    q: &[f32],
    preds: &[Pred],
    exclude_pointed_by: &[String],
    identity: &str,
) -> Result<(Vec<(f32, usize, usize)>, usize)> {
    let mut hits: Vec<(f32, usize, usize)> = Vec::new();
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
                hits.push((dot(q, &floats[slot * dim..(slot + 1) * dim]), owner, slot));
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
                && fm
                    .get(identity)
                    .is_some_and(|v| pointed.contains(&scalar_text(v)))
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
                hits.push((dot(q, &floats[slot * dim..(slot + 1) * dim]), owner, *slot));
            }
        }
    }
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

/// How a reference prints.
///
/// One root prints the path the caller already knows the root of. Several print
/// which root the section came from, because two indexes may hold the same
/// relative path and `--paths-only` exists for a caller that opens what it is
/// handed. Relative to the working directory where it can be, so that the
/// common case reads like the one-root case.
fn reference(roots: &[PathBuf], owner: usize, rel: &str) -> String {
    if roots.len() == 1 {
        return rel.to_string();
    }
    let joined = roots[owner].join(rel);
    std::env::current_dir()
        .ok()
        .and_then(|cwd| joined.strip_prefix(cwd).ok().map(|p| p.display().to_string()))
        .unwrap_or_else(|| joined.display().to_string())
}

fn cmd_query(
    roots: &[PathBuf],
    text: &str,
    wheres: &[String],
    exclude_pointed_by: &[String],
    identity: &str,
    limit: usize,
    paths_only: bool,
    refresh: bool,
) -> Result<()> {
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
        if q.is_none() {
            q = Some(embed_and_prove(&opened, text)?);
        }
        let q = q.as_deref().expect("embedded above");

        let (hits, dropped) = rank(&opened, q, &preds, exclude_pointed_by, identity)?;
        // Hydrated wider than the limit, because merging joins pieces that both
        // rank and the limit counts merged rows rather than sections.
        // Never below the limit: a caller asking for every record must be able
        // to receive every record, which D-01M1PP6HGWJMQC's fence checks.
        let window = (limit * MERGE_WINDOW).max(32);
        let top: Vec<(f32, usize, usize)> = hits.iter().take(window).copied().collect();

        // Hydrated one index at a time and put back in the order they ranked,
        // because a slot means a row in one index's matrix.
        let mut hydrated: Vec<Option<Section>> = vec![None; top.len()];
        for owner in 0..opened.len() {
            let mut at: Vec<usize> = Vec::new();
            let mut want: Vec<usize> = Vec::new();
            for (i, (_, ow, slot)) in top.iter().enumerate() {
                if *ow == owner {
                    at.push(i);
                    want.push(*slot);
                }
            }
            if want.is_empty() {
                continue;
            }
            for (i, sec) in at.into_iter().zip(opened[owner].st.hydrate(&want)?) {
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
        for (i, (score, owner, _)) in top.iter().enumerate() {
            let sec = hydrated[i].take().expect("every ranked slot hydrated");
            let key = (named[*owner].join(&sec.path), sec.start, sec.end);
            if !seen.insert(key) {
                held += 1;
                continue;
            }
            rows.push(sec);
            owners.push(*owner);
            scores.push(*score);
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
                if paths_only { eprintln!("{notice}") } else { println!("{notice}") }
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
            let notice =
                format!("({held} section(s) held by more than one of these indexes, returned once)");
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
            if paths_only { eprintln!("{said}") } else { println!("{said}") }
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

/// How many sections an index commits at a time. A run that dies keeps every
/// batch before the one it was in, so this is what a kill costs: at the 0.53 s
/// per 32 inputs measured on 2026-09-06, about eight seconds of embedding.
const FLUSH: usize = 512;

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
            let touches = run
                .last()
                .is_some_and(|&last| rows[last].end + 1 == rows[i].start);
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
fn trail_of(s: &Section) -> String {
    let mut trail = s.breadcrumb.clone();
    trail.push(s.heading.clone().unwrap_or_else(|| "(preamble)".to_string()));
    trail.join(" > ")
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
/// docs/measurements/ was measured on. What folio does know is the port its
/// own configuration points at, which is the part that is easy to get wrong.
/// The command each backend needs, and the flags folio's own accounting rests
/// on rather than the caller's taste.
///
/// `--auto-truncate false` is not a preference. Measured 2026-09-07: with TEI's
/// default a section past the model's input length comes back as a vector of its
/// beginning with a 200, so folio's budget calibration — which works by sending
/// its longest section and watching for a refusal — sees nothing, and the index
/// records that nothing was cut. So it is printed always, and there is no flag
/// to turn it off.
fn backend_args(
    backend: Backend,
    hf: &str,
    hf_file: &str,
    pooling: &str,
    context: usize,
    host: String,
    port: u16,
) -> Vec<String> {
    let mut args: Vec<String> = match backend {
        Backend::LlamaCpp => vec![
            "--embeddings".into(),
            "-hf".into(), hf.into(),
            "--hf-file".into(), hf_file.into(),
            "--pooling".into(), pooling.into(),
            "-c".into(), context.to_string(),
            "-b".into(), context.to_string(),
            "-ub".into(), context.to_string(),
        ],
        Backend::Tei => vec![
            "--model-id".into(), hf.into(),
            "--auto-truncate".into(), "false".into(),
            "--max-batch-tokens".into(), context.to_string(),
        ],
    };
    args.extend(["--host".to_string(), host, "--port".to_string(), port.to_string()]);
    args
}

fn cmd_unit(
    root: &Path,
    backend: Backend,
    launchd: bool,
    systemd: bool,
    hf: Option<&str>,
    hf_file: Option<&str>,
    pooling: Option<&str>,
    context: usize,
) -> Result<()> {
    // Refused rather than dropped. A unit that quietly ignored a flag would be
    // the failure this command exists to prevent, one step earlier.
    if backend == Backend::Tei {
        for (flag, given) in [("--hf-file", hf_file.is_some()), ("--pooling", pooling.is_some())] {
            if given {
                bail!(
                    "{flag} is llama-server's and means nothing to text-embeddings-router, \
                     which reads safetensors and takes pooling from the model"
                );
            }
        }
    }
    let (server_name, hf) = match backend {
        Backend::LlamaCpp => (
            "llama-server",
            hf.unwrap_or("keisuke-miyako/gte-modernbert-base-gguf"),
        ),
        Backend::Tei => (
            "text-embeddings-router",
            hf.unwrap_or("Alibaba-NLP/gte-modernbert-base"),
        ),
    };
    let hf_file = hf_file.unwrap_or("gte-modernbert-base-Q8_0.gguf");
    let pooling = pooling.unwrap_or("cls");
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
    let server = which_server(server_name);
    let args = backend_args(backend, hf, hf_file, pooling, context, host, port);

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
     {server_name} cannot be started on demand, because it binds its own socket
     rather than accepting one from launchd. It is not free while idle:
     docs/measurements/ in the folio repository carries what it costs. -->
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
             # {server_name} cannot be socket-activated, because it binds its own\n\
             # socket rather than accepting one from systemd. It is not free while\n\
             # idle: docs/measurements/ in the folio repository carries the cost.\n\
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
fn which_server(name: &str) -> String {
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
            return Err(e.into());
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
                anyhow::Error::from(e).to_string().lines().next().unwrap_or_default()
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
            println!("  structure  could not measure: {}", anyhow::Error::from(e));
        }
        Ok(v) => {
            let (near, far) = (dot(&v[0], &v[1]), dot(&v[0], &v[2]));
            print!("  structure  paraphrase {near:.3}, unrelated {far:.3}");
            // ponytail: two thresholds on one English triple. It catches an
            // inverted or collapsed space, not a merely mediocre one; a real
            // quality measure is the unmeasured question in docs/measurements/.
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

fn cmd_status(roots: &[PathBuf], list_truncated: bool) -> Result<()> {
    for (i, root) in roots.iter().enumerate() {
        if i > 0 {
            println!();
        }
        status_of(root, list_truncated)?;
    }
    Ok(())
}

fn status_of(root: &Path, list_truncated: bool) -> Result<()> {
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

    // Listing answers a different question from the report, and reads only the
    // rows it names rather than every record.
    if list_truncated {
        let cut = st.truncated()?;
        if cut.is_empty() {
            println!("no section was cut at {} characters", meta.max_chars);
            return Ok(());
        }
        println!(
            "{} of {} sections were cut at {} characters, and ranked on what was left:",
            cut.len(),
            counts.sections,
            meta.max_chars
        );
        for sec in &cut {
            println!("  {}:{}-{}", sec.path, sec.start, sec.end);
            println!("        {}", trail_of(sec));
        }
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
    fn a_tei_unit_carries_the_flag_folio_cannot_check_at_runtime() {
        let args = backend_args(Backend::Tei, "org/model", "ignored", "ignored", 8192,
                                "127.0.0.1".into(), 8080);
        let pairs: Vec<(&str, &str)> = args
            .windows(2)
            .map(|w| (w[0].as_str(), w[1].as_str()))
            .collect();
        // Without this the endpoint answers 200 with a vector of a section's
        // beginning and folio's budget calibration has nothing to react to.
        assert!(pairs.contains(&("--auto-truncate", "false")));
        assert!(pairs.contains(&("--max-batch-tokens", "8192")));
        assert!(pairs.contains(&("--model-id", "org/model")));
        // llama-server's flags mean nothing here and are not printed anyway.
        for absent in ["--hf-file", "--pooling", "-ub", "--embeddings"] {
            assert!(!args.iter().any(|a| a == absent), "{absent} reached a TEI unit");
        }
    }

    #[test]
    fn the_default_backend_prints_what_it_always_printed() {
        let args = backend_args(Backend::LlamaCpp, "repo", "file.gguf", "cls", 8192,
                                "127.0.0.1".into(), 8080);
        assert_eq!(
            args,
            ["--embeddings", "-hf", "repo", "--hf-file", "file.gguf", "--pooling", "cls",
             "-c", "8192", "-b", "8192", "-ub", "8192", "--host", "127.0.0.1",
             "--port", "8080"]
        );
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
        let current = HashMap::from([
            ("a.md".to_string(), stamped(7)),
            ("b.md".to_string(), stamped(7)),
        ]);
        let changed = HashSet::from(["b.md".to_string()]);
        assert!(pair_moves(&prev, &current, &changed).is_empty());
    }

    #[test]
    fn one_departure_supplies_only_one_of_two_arrivals() {
        let prev = HashMap::from([("a.md".to_string(), stamped(7))]);
        let current = HashMap::from([
            ("b.md".to_string(), stamped(7)),
            ("c.md".to_string(), stamped(7)),
        ]);
        let changed = HashSet::from(["b.md".to_string(), "c.md".to_string()]);
        let moves = pair_moves(&prev, &current, &changed);
        assert_eq!(moves, vec![("a.md".to_string(), "b.md".to_string())]);
    }

    #[test]
    fn a_different_dimension_is_a_different_space_and_not_a_near_miss() {
        let then = vec![1.0, 0.0, 0.0];
        assert_eq!(same_space(&[1.0, 0.0, 0.0], &then), 1.0);
        assert_eq!(same_space(&[1.0, 0.0], &then), 0.0, "a shorter vector must not score on the dimensions it shares");
        assert_eq!(same_space(&[1.0, 0.0, 0.0], &[]), 0.0, "an index with no fingerprint is not a match");
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
        assert!(!keeps(&fm(json!({"status": "draft", "type": "note"})), &pred),
                "the second flag has to still be required");
        assert!(!keeps(&fm(json!({"status": "retired", "type": "guide"})), &pred));
    }

    #[test]
    fn a_bare_key_beside_an_or_is_hinted_at_and_not_refused() {
        // The precedence reading of `status=live|draft` is `status=live` or the
        // presence of a key `draft`, which is rarely meant and is still a real
        // thing to ask for. So it works, and it says so.
        let (pred, hints) = parse(&["status=live|draft"]);
        assert!(keeps(&fm(json!({"draft": true})), &pred),
                "the presence test has to keep working");
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("presence test for the key `draft`"), "{}", hints[0]);
        assert!(parse(&["status=live | status=draft"]).1.is_empty(),
                "an or of two comparisons needs no hint");
    }

    #[test]
    fn the_predicate_that_emptied_a_result_is_named() {
        let rows = vec![(0, fm(json!({"status": "live"}))), (1, fm(json!({"status": "retired"})))];
        let pred = preds(&["status=live | status=draft", "type=guide"]);
        assert_eq!(blames(&rows, &pred), vec!["type=guide".to_string()],
                   "only the predicate that kept nothing on its own is named");
        assert!(blames(&rows, &preds(&["status=live"])).is_empty());
    }

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
    fn a_reference_names_its_root_only_where_there_are_several() {
        let one = [PathBuf::from("/corpus")];
        assert_eq!(reference(&one, 0, "docs/a.md"), "docs/a.md");
        let two = [PathBuf::from("/corpus"), PathBuf::from("/other")];
        assert_eq!(reference(&two, 1, "docs/a.md"), "/other/docs/a.md");
    }

    #[test]
    fn key_not_equal_still_passes_when_the_key_is_absent() {
        // The documented behaviour, kept: a filter must not silently drop the
        // documents nobody has annotated yet.
        assert!(keeps(&fm(json!({"title": "Beta"})), &preds(&["status!=deprecated"])));
    }
}

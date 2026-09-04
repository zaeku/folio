//! folio — section-scoped semantic search over a markdown corpus.
//!
//! The index holds references (path, line range, frontmatter), never bodies.
//! Retrieval names sections to read; reading the file is a separate step, so a
//! stale index costs a wasted candidate and never a wrong quotation.

mod sections;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use sections::Section;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::{DefaultHasher, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const DIR: &str = ".folio";
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
        #[arg(long, default_value_t = 12_000)]
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
            limit,
        } => cmd_query(&root, &text, &wheres, limit),
        Cmd::Status { root } => cmd_status(&root),
    }
}

// ---------------------------------------------------------------- persistence

#[derive(Serialize, Deserialize, Default)]
struct State {
    model: String,
    endpoint: String,
    dim: usize,
    /// Path to content hash, so only changed files are re-embedded. No file
    /// watcher: the files are the truth and hashing them is cheap.
    files: HashMap<String, u64>,
}

type Row = (Section, Vec<f32>);

fn store(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let d = root.join(DIR);
    (
        d.join("sections.jsonl"),
        d.join("vectors.f32"),
        d.join("state.json"),
    )
}

fn load(root: &Path) -> Result<(State, Vec<Row>)> {
    let (jsonl, vecs, state) = store(root);
    if !state.exists() {
        return Ok((State::default(), Vec::new()));
    }
    let state: State = serde_json::from_slice(&fs::read(&state)?)
        .with_context(|| format!("{} is unreadable", state.display()))?;
    let dim = state.dim;
    if dim == 0 {
        return Ok((state, Vec::new()));
    }

    let secs: Vec<Section> = BufReader::new(fs::File::open(&jsonl)?)
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_>>()
        .with_context(|| format!("{} is unreadable", jsonl.display()))?;

    let raw = fs::read(&vecs)?;
    let floats: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if floats.len() != secs.len() * dim {
        bail!(
            "index is inconsistent: {} sections but {} floats at dim {} — rerun with --rebuild",
            secs.len(),
            floats.len(),
            dim
        );
    }

    Ok((
        state,
        secs.into_iter()
            .zip(floats.chunks_exact(dim))
            .map(|(s, v)| (s, v.to_vec()))
            .collect(),
    ))
}

fn save(root: &Path, state: &State, rows: &[Row]) -> Result<()> {
    let (jsonl, vecs, state_path) = store(root);
    fs::create_dir_all(root.join(DIR))?;

    let mut w = BufWriter::new(fs::File::create(&jsonl)?);
    for (s, _) in rows {
        serde_json::to_writer(&mut w, s)?;
        w.write_all(b"\n")?;
    }
    w.flush()?;

    let mut w = BufWriter::new(fs::File::create(&vecs)?);
    for (_, v) in rows {
        for x in v {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    w.flush()?;

    fs::write(&state_path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
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
        let resp: EmbedResp = ureq::post(endpoint)
            .send_json(EmbedReq { model, input: batch })
            .with_context(|| format!("embedding request to {endpoint} failed"))?
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
    match v {
        Value::String(s) => s == want,
        other => other.to_string().trim_matches('"') == want,
    }
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
        (State::default(), Vec::new())
    } else {
        load(&root)?
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

    let mut current: HashMap<String, u64> = HashMap::new();
    let mut changed: HashSet<String> = HashSet::new();
    for entry in ignore::WalkBuilder::new(&root).build() {
        let entry = entry?;
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let rel = path
            .strip_prefix(&root)?
            .to_string_lossy()
            .replace('\\', "/");
        let h = hash(&fs::read(path)?);
        if !reuse || prev.files.get(&rel) != Some(&h) {
            changed.insert(rel.clone());
        }
        current.insert(rel, h);
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
        &State {
            model: model.to_string(),
            endpoint: endpoint.to_string(),
            dim,
            files: current,
        },
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

fn cmd_query(root: &Path, text: &str, wheres: &[String], limit: usize) -> Result<()> {
    let root = root.canonicalize()?;
    let (state, rows) = load(&root)?;
    if rows.is_empty() {
        bail!("no index under {} — run `folio index` first", root.display());
    }

    let q = embed(&state.endpoint, &state.model, &[text.to_string()])?
        .pop()
        .expect("one input yields one vector");

    let preds = parse_preds(wheres);
    let mut hits: Vec<(f32, &Section)> = rows
        .iter()
        .filter(|(s, _)| keeps(s, &preds))
        .map(|(s, v)| (dot(&q, v), s))
        .collect();
    hits.sort_by(|a, b| b.0.total_cmp(&a.0));

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
    let (state, rows) = load(&root)?;
    if rows.is_empty() {
        println!("no index under {}", root.display());
        return Ok(());
    }
    let keys: HashSet<&String> = rows.iter().flat_map(|(s, _)| s.fm.keys()).collect();
    let mut keys: Vec<&&String> = keys.iter().collect();
    keys.sort();

    println!("{}", root.display());
    println!("  files       {}", state.files.len());
    println!("  sections    {}", rows.len());
    println!("  truncated   {}", rows.iter().filter(|(s, _)| s.truncated).count());
    println!("  model       {} @ {}", state.model, state.endpoint);
    println!("  dim         {}", state.dim);
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

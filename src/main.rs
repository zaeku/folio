//! folio — section-scoped semantic search over a markdown corpus.
//!
//! The index holds references (path, line range, frontmatter), never bodies.
//! Retrieval names sections to read; reading the file is a separate step, so a
//! stale index costs a wasted candidate and never a wrong quotation.

mod cli;
mod config;
mod embed;
mod filters;
mod index;
mod query;
mod sections;
mod store;
mod unit;

use anyhow::{Context, Result, bail};
use clap::Parser;

use cli::{Cli, Cmd, ConfigCmd};
use config::{
    DEFAULT_ENDPOINT, DEFAULT_MODEL, PROJECT_CONFIG, config_path, corpus_root, nearest_declaration,
    read_config_at, resolve, resolve_allow_insecure, resolve_api_key,
};
use embed::{dot, embed};
use index::{FLUSH, append, cmd_index, hash};
use query::{cmd_query, trail_of};
use sections::Section;
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;
use store::{Meta, Stamp, Store};
use unit::cmd_unit;

/// What a repeatable `--root` means when it was not given.
fn roots_or_enclosing(named: Vec<PathBuf>) -> Vec<PathBuf> {
    if named.is_empty() {
        return vec![corpus_root()];
    }
    named
}

fn main() -> Result<()> {
    // A command that is piped into `head` has its stdout closed early. Rust
    // ignores SIGPIPE, so the next `println!` panics and prints a backtrace
    // notice over the output the reader was already reading. Restoring the
    // default ends this process the way every other command in a pipeline
    // ends.
    unsafe { libc::signal(libc::SIGPIPE, libc::SIG_DFL) };

    match Cli::parse().cmd {
        Cmd::Index { root, endpoint, model, allow_insecure, max_chars, rebuild } => cmd_index(
            &root,
            endpoint.as_deref(),
            model.as_deref(),
            allow_insecure,
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
            allow_insecure,
        } => cmd_query(
            &roots_or_enclosing(root),
            &text,
            &wheres,
            &exclude_pointed_by,
            &identity,
            limit,
            paths_only,
            !no_refresh,
            allow_insecure,
        ),
        Cmd::Config { action, root } => cmd_config(action, &root),
        Cmd::Doctor { root, endpoint, model, allow_insecure, max_chars } => cmd_doctor(
            &root.unwrap_or_else(corpus_root),
            endpoint.as_deref(),
            model.as_deref(),
            allow_insecure,
            max_chars,
        ),
        Cmd::Unit { root, backend, launchd, systemd, hf, hf_file, pooling, context } => cmd_unit(
            &root,
            backend,
            launchd,
            systemd,
            hf.as_deref(),
            hf_file.as_deref(),
            pooling.as_deref(),
            context,
        ),
        Cmd::Extract { prefix, into, root } => {
            cmd_extract(&root.unwrap_or_else(corpus_root), &prefix, &into)
        }
        Cmd::Skill => {
            // The same bytes as the published file, because they are the
            // published file. A copy typed in here could disagree with it and
            // nothing would notice.
            print!("{}", include_str!("../skill/SKILL.md"));
            Ok(())
        }
        Cmd::Status { root, truncated } => cmd_status(&roots_or_enclosing(root), truncated),
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
    if !bytes.is_multiple_of(meta.dim * 4) || bytes / (meta.dim * 4) < last {
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

// ------------------------------------------------------------------- commands

/// What a prefix means: the subtree it names, and not the names it starts.
///
/// `keep` takes `keep/a.md` and leaves `keeper.md`, which is the whole reason
/// this is a function rather than a `starts_with` at the call site.
fn subtree(prefix: &str) -> String {
    format!("{}/", prefix.trim_matches('/'))
}

/// Copy the rows under `prefix` into an index of their own.
///
/// Nothing is embedded and no endpoint is called: the vectors already exist and
/// the meta they belong to travels with them, fingerprint included, so the
/// slice is in its source's space by construction rather than by assertion.
fn cmd_extract(root: &Path, prefix: &str, into: &Path) -> Result<()> {
    let root = root.canonicalize().with_context(|| format!("{} not found", root.display()))?;
    let into = into.canonicalize().with_context(|| {
        format!(
            "{} not found — a slice is written beside files that are already there",
            into.display()
        )
    })?;
    if root == into {
        bail!("a corpus cannot be extracted into itself");
    }
    if store::db_path(&into).exists() {
        bail!(
            "{} already holds an index — a slice is written into a corpus that has none, and merging two is not this command",
            into.display()
        );
    }

    let Some((src, meta)) = open_index(&root)? else {
        bail!("no index under {} — run `folio index` first", root.display());
    };
    let map = map_vectors(&root)?.context("the index has records but no vectors")?;
    let floats = as_floats(&map)?;

    let under = subtree(prefix);
    let slots = src.slots()?;
    let rows = src.hydrate(&slots)?;
    let taken: Vec<(usize, Section)> =
        slots.into_iter().zip(rows).filter(|(_, s)| s.path.starts_with(&under)).collect();
    if taken.is_empty() {
        bail!("no section under {prefix} in {}", root.display());
    }

    // Every row's file has to be at the destination, and be the file the index
    // recorded. The hash is already stored per file, so this is exact rather
    // than a guess from paths, and it runs before anything is written.
    let stamps = src.files()?;
    let mut missing: Vec<String> = Vec::new();
    let mut carried: Vec<(String, Stamp)> = Vec::new();
    for path in taken.iter().map(|(_, s)| &s.path).collect::<BTreeSet<_>>() {
        let rel = path.strip_prefix(&under).expect("filtered on this prefix");
        let Some(was) = stamps.get(path) else {
            missing.push(format!("{rel} (the index recorded no hash for it)"));
            continue;
        };
        match fs::read(into.join(rel)) {
            Ok(bytes) if hash(&bytes) == was.hash => carried.push((rel.to_string(), *was)),
            Ok(_) => missing.push(format!("{rel} (holds different text)")),
            Err(e) => missing.push(format!("{rel} ({e})")),
        }
    }
    if let Some(first) = missing.first() {
        bail!(
            "{} of the files this slice names are not at {} as the index recorded them, starting with {first}",
            missing.len(),
            into.display()
        );
    }

    let dst = Store::open(&into)?;
    let mut renumbered: Vec<(usize, Section)> = Vec::new();
    for (slot, (_, section)) in taken.iter().enumerate() {
        let mut section = section.clone();
        section.path = section.path.strip_prefix(&under).expect("filtered").to_string();
        renumbered.push((slot, section));
    }
    dst.add(&renumbered)?;
    dst.stamp(carried.iter().map(|(p, s)| (p, s)))?;
    dst.set_meta(&meta)?;

    // Written in the order the records name, and in batches for the same reason
    // an index is: a slice of a large corpus is a large matrix.
    let dim = meta.dim;
    for (batch, chunk) in taken.chunks(FLUSH).enumerate() {
        let base = batch * FLUSH;
        let vectors: Vec<Vec<f32>> =
            chunk.iter().map(|(slot, _)| floats[slot * dim..(slot + 1) * dim].to_vec()).collect();
        append(&into, base, dim, &vectors)?;
    }

    println!(
        "extracted {} files · {} sections into {}",
        carried.len(),
        taken.len(),
        into.display()
    );
    println!("  model       {} @ {}", meta.model, meta.endpoint);
    Ok(())
}

fn cmd_config(action: Option<ConfigCmd>, root: &Path) -> Result<()> {
    let user_path = config_path();
    let proj_path = root.join(PROJECT_CONFIG);

    if let Some(ConfigCmd::Set { key, value, project }) = action {
        if key == "api_key" && project {
            bail!(
                "a project config cannot hold an API key because folio.yaml is committed with the corpus; \
                 write to user config without --project or set FOLIO_API_KEY in the environment"
            );
        }
        let path = if project { proj_path.clone() } else { user_path.clone() };
        let mut cfg = read_config_at(&path)?;
        match key.as_str() {
            "endpoint" => cfg.endpoint = Some(value),
            "model" => cfg.model = Some(value),
            "api_key" => cfg.api_key = Some(value),
            "allow_insecure" => {
                cfg.allow_insecure =
                    Some(value.parse().context("allow_insecure must be 'true' or 'false'")?)
            }
            other => bail!(
                "no setting named {other} — folio config holds endpoint, model, api_key, allow_insecure"
            ),
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
    let (model, m_src) =
        resolve(None, "FOLIO_MODEL", proj.model.as_deref(), user.model.as_deref(), DEFAULT_MODEL);
    let (_api_key, k_src) = resolve_api_key(&endpoint, user.api_key.as_deref());
    println!("  endpoint  {endpoint}  ({e_src})");
    println!("  model     {model}  ({m_src})");
    if k_src != "none" {
        println!("  api_key   [configured]  ({k_src})");
    }
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
    allow_insecure: bool,
    max_chars: usize,
) -> Result<()> {
    let declared = nearest_declaration(root).unwrap_or_else(|| root.join(PROJECT_CONFIG));
    let proj = read_config_at(&declared)?;
    let user = read_config_at(&config_path())?;
    let (endpoint, e_src) = resolve(
        endpoint,
        "FOLIO_ENDPOINT",
        proj.endpoint.as_deref(),
        user.endpoint.as_deref(),
        DEFAULT_ENDPOINT,
    );
    let (model, _) =
        resolve(model, "FOLIO_MODEL", proj.model.as_deref(), user.model.as_deref(), DEFAULT_MODEL);
    let (api_key, k_src) = resolve_api_key(&endpoint, user.api_key.as_deref());
    let allow_insecure = resolve_allow_insecure(allow_insecure, user.allow_insecure);
    println!("  endpoint   {endpoint}  ({e_src})");
    println!("  model      {model}");
    if k_src != "none" {
        println!("  api_key    [configured]  ({k_src})");
    }

    let mut failed = false;

    // 1. It answers, and with how many dimensions.
    let t = std::time::Instant::now();
    let probe = match embed(
        &endpoint,
        &model,
        api_key.as_deref(),
        allow_insecure,
        &["a sentence to embed".to_string()],
    ) {
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
    match embed(&endpoint, &model, api_key.as_deref(), allow_insecure, std::slice::from_ref(&long))
    {
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
    match embed(&endpoint, &model, api_key.as_deref(), allow_insecure, &probes) {
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
                println!(
                    "             check the server's pooling: this model family wants one \
                          specific mode, and the default is wrong for some of them"
                );
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
    // A corpus that arrived from another machine carries an index built in one
    // space and a declaration naming another. `folio index` is where the two
    // meet, and it meets them by re-embedding everything.
    if let Some(path) = nearest_declaration(&root) {
        let decl = read_config_at(&path)?;
        let differs = decl.model.as_deref().is_some_and(|m| m != meta.model)
            || decl.endpoint.as_deref().is_some_and(|e| e != meta.endpoint);
        if differs {
            println!(
                "  declared    {} @ {}  ({})",
                decl.model.as_deref().unwrap_or(&meta.model),
                decl.endpoint.as_deref().unwrap_or(&meta.endpoint),
                path.display()
            );
        }
    }
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

    /// The parser folio publishes, asked to check itself.
    ///
    /// `debug_assert` is clap's own audit of a built `Command`, and it runs
    /// nowhere else: nothing in this binary constructs the parser except
    /// `main`, so a `Cli` that clap would reject at startup compiles, passes
    /// every other test, and passes CI. What it caught when it was written was
    /// a flag whose `#[arg(long)]` had drifted onto the field below it, which
    /// left the flag as a positional taking no value.
    ///
    /// The invocations beneath it are the surface folio's own sentences name.
    /// An error message that tells a caller to rerun `folio index --rebuild` is
    /// a promise that the flag exists, and this is where that promise is kept.
    #[test]
    fn a_prefix_names_a_subtree_and_not_the_names_it_starts() {
        let under = subtree("keep");
        assert!("keep/a.md".starts_with(&under));
        assert!("keep/deep/b.md".starts_with(&under));
        assert!(!"keeper.md".starts_with(&under));
        assert!(!"keep.md".starts_with(&under));
        // A caller who writes the separator means the same subtree.
        assert_eq!(subtree("keep/"), under);
        assert_eq!(subtree("/keep/"), under);
    }

    #[test]
    fn the_published_command_surface_holds() {
        use clap::CommandFactory;
        Cli::command().debug_assert();

        for argv in [
            vec!["folio", "index", "--rebuild"],
            vec!["folio", "index", "--max-chars", "4000"],
            vec!["folio", "query", "a question", "--paths-only"],
            vec!["folio", "query", "a question", "--where", "status=live", "--no-refresh"],
            vec!["folio", "status", "--truncated"],
            vec!["folio", "doctor"],
            vec!["folio", "unit", "--systemd"],
            vec!["folio", "config", "set", "endpoint", "http://127.0.0.1:8080/v1/embeddings"],
            vec!["folio", "skill"],
        ] {
            Cli::try_parse_from(&argv).unwrap_or_else(|e| {
                panic!("{} is documented and did not parse: {e}", argv.join(" "))
            });
        }
    }
}

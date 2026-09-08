//! The command line folio publishes.
//!
//! Declaration and help text, and no work: what a flag means to a caller is
//! written here, and what it makes folio do is written where that happens. The
//! sentences below are the ones `--help` prints, so they are the interface as
//! much as the names are.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::config::{MAX_CHARS, at_least_one};
use crate::unit::Backend;

#[derive(Parser)]
#[command(name = "folio", version, about, long_about = None)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}
#[derive(Subcommand)]
pub(crate) enum Cmd {
    /// Index every markdown file under `root`, re-embedding only changed files.
    Index {
        /// The corpus to index: every `.md` file below it, and `.folio/`
        /// beside it.
        ///
        /// Taken as given. A query walks upward to find the corpus it is
        /// standing in; this does not.
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
        /// Allow sending credentials over unencrypted HTTP off the loopback.
        #[arg(long)]
        allow_insecure: bool,
    },
    /// Rank sections by meaning, optionally filtered on frontmatter.
    Query {
        /// The question, in whatever words you have.
        ///
        /// It is embedded and ranked by meaning, so it carries no operators and
        /// shares nothing with an `rg` pattern.
        text: String,
        /// The corpus to rank. Repeatable.
        ///
        /// Several roots are ranked as one answer, and must be one vector
        /// space: the same weights, proven by the fingerprint each index
        /// carries, and the same character budget. A root with no index, or one
        /// folio cannot show belongs with the others, is refused rather than
        /// left out. They may cover the same files; a section two of them hold
        /// is returned once, and the query says how many it collapsed.
        ///
        /// Given none, folio answers from the corpus enclosing the working
        /// directory: it walks upward to the first `.folio/` or `folio.yaml`
        /// and uses that, so a question can be asked from anywhere inside a
        /// corpus.
        #[arg(long)]
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
        /// Allow sending credentials over unencrypted HTTP off the loopback.
        #[arg(long)]
        allow_insecure: bool,
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
        ///
        /// Given none, the corpus enclosing the working directory.
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        endpoint: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// The section budget to test the endpoint's batch size against.
        #[arg(long, default_value_t = MAX_CHARS, value_parser = at_least_one)]
        max_chars: usize,
        /// Allow sending credentials over unencrypted HTTP off the loopback.
        #[arg(long)]
        allow_insecure: bool,
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
    /// Copy part of an index into a corpus of its own. Embeds nothing.
    ///
    /// The destination must already hold the files: folio moves references and
    /// vectors, and the text is moved by whatever moves text for you. Every
    /// row's file is checked against the hash the index recorded for it, and a
    /// slice carries its source's model, budget and fingerprint, so a query can
    /// read the two together and either can be proven against an endpoint.
    Extract {
        /// The path prefix to take, relative to the corpus root.
        prefix: String,
        /// The corpus to write the index into. It must hold the files already
        /// and must not hold an index.
        #[arg(long)]
        into: PathBuf,
        /// The corpus to cut from.
        ///
        /// Given none, the corpus enclosing the working directory.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Print folio's skill document, the guidance written for an agent using it.
    ///
    /// It is the file the crate publishes, embedded at compile time. It is
    /// printed and not installed: where it belongs is the caller's to decide,
    /// the way `folio unit` leaves a service file to whoever manages services.
    Skill,
    /// Report what the index covers.
    Status {
        /// The corpus to report on. Repeatable, one block each.
        ///
        /// Given none, the corpus enclosing the working directory.
        root: Vec<PathBuf>,
        /// List the sections that were cut to the budget, rather than counting
        /// them. Each was ranked on part of its text.
        #[arg(long)]
        truncated: bool,
    },
}
#[derive(Subcommand)]
pub(crate) enum ConfigCmd {
    /// Write one setting to a config file, creating it if it is absent.
    Set {
        /// `endpoint`, `model`, `api_key`, or `allow_insecure`.
        key: String,
        value: String,
        /// Write `folio.yaml` beside the corpus instead of the user's config.
        /// Commit that file, and everyone who indexes the corpus embeds it the
        /// same way.
        #[arg(long)]
        project: bool,
    },
}

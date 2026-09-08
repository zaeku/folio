//! Where a setting comes from, which corpus a command is standing in, and what
//! may be sent where.
//!
//! A flag beats the environment, the environment beats the corpus's own
//! `folio.yaml`, and that beats the user's. A declaration is inherited by a
//! subtree and an index is not — D-01M200KPGVT37G — and a credential travels to
//! the loopback in plaintext and off it only over TLS — D-01M1ZWJ59967PE.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

use crate::store;

/// The default character budget per section. Set below the 8192-token context
/// of the models folio is measured on rather than at a round number.
pub(crate) const MAX_CHARS: usize = 8_000;
/// Where folio looks for an endpoint when nothing else names one.
pub(crate) const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8080/v1/embeddings";
/// The model name folio sends. Most single-model servers ignore it.
pub(crate) const DEFAULT_MODEL: &str = "default";

/// What `folio index` needs before an index exists to remember it.
///
/// `folio query` never reads this. An index records the endpoint and model it
/// was built with, and a vector space belongs to one of each, so the recorded
/// pair is the only correct answer for a corpus that has one.
#[derive(Debug, Default, Deserialize, Serialize)]
pub(crate) struct Config {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) endpoint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) api_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) allow_insecure: Option<bool>,
}

/// The per-corpus config, meant to be committed with the corpus.
///
/// It sits beside the corpus rather than inside `.folio/`, because `.folio/` is
/// a derived index that belongs to whoever built it. Which model a corpus needs
/// is not derived: a Korean corpus and an English one can want different ones,
/// and everyone who indexes that corpus wants the same answer.
pub(crate) const PROJECT_CONFIG: &str = "folio.yaml";

pub(crate) fn config_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".config")
    });
    base.join("folio").join("config.yaml")
}

/// The corpus a command answers from when none was named.
///
/// Upward from `start` to the first directory holding either `.folio/` or
/// `folio.yaml`, so a question can be asked from anywhere inside a corpus. One
/// stat per directory between the two, which is not the downward discovery
/// D-01M1X1GGMV5001 declines: that one walks a corpus to find nested indexes
/// and costs 172 ms over 14,616 files against a 120 ms query, measured
/// 2026-09-05.
///
/// `folio.yaml` stops it as well as `.folio/`, and stops it for the reason
/// cargo stops at `Cargo.toml` rather than at `target/`: the index is derived
/// and disposable while the model a corpus needs is not. A project that
/// declares a corpus it has not indexed yet therefore ends the walk and is
/// named in the refusal, instead of being passed on the way to an unrelated
/// index somewhere above it.
pub(crate) fn corpus_root_from(start: &Path) -> Option<PathBuf> {
    start
        .ancestors()
        .find(|dir| dir.join(store::DIR).exists() || dir.join(PROJECT_CONFIG).exists())
        .map(Path::to_path_buf)
}

/// The `folio.yaml` that governs `dir`: the nearest one at or above it.
///
/// A declaration belongs to the corpus rather than to the directory a command
/// was run in, so a subtree of a corpus embeds with the model that corpus
/// chose. Only `folio.yaml` ends this walk. An index is not inherited: what one
/// holds is settled by the fingerprint it recorded, never by a name in a file
/// above it.
pub(crate) fn nearest_declaration(dir: &Path) -> Option<PathBuf> {
    // Absolute first: `.` has no ancestors but itself, and a root is usually
    // given as `.`, so a relative path would end this walk before it started.
    let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    dir.ancestors().map(|d| d.join(PROJECT_CONFIG)).find(|p| p.exists())
}

/// The index above `root`, when `root` is indexed inside another corpus.
///
/// Strictly above: an index at `root` is the one about to be updated.
pub(crate) fn enclosing_index(root: &Path) -> Option<PathBuf> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    root.ancestors().skip(1).find(|d| d.join(store::DIR).is_dir()).map(Path::to_path_buf)
}

/// The same, from the working directory, falling back to it.
///
/// A caller standing outside any corpus gets what they always got: the working
/// directory, and the refusal that names it.
pub(crate) fn corpus_root() -> PathBuf {
    std::env::current_dir()
        .ok()
        .and_then(|cwd| corpus_root_from(&cwd))
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Read the config file, or fail.
///
/// A file that does not parse is not the same as no file. Falling back to the
/// default endpoint would index a corpus against a model the user did not
/// choose, and vectors from the wrong model are not detectable from a ranking.
pub(crate) fn read_config_at(path: &Path) -> Result<Config> {
    match fs::read_to_string(path) {
        Ok(text) => {
            let cfg: Config = serde_yaml_ng::from_str(&text)
                .with_context(|| format!("{} is not valid YAML", path.display()))?;
            if path.file_name().is_some_and(|n| n == PROJECT_CONFIG) && cfg.api_key.is_some() {
                bail!(
                    "{} contains an API key; folio.yaml is committed with the corpus and must not \
                     contain credentials — use FOLIO_API_KEY in the environment or write to user config \
                     with `folio config set api_key <val>`",
                    path.display()
                );
            }
            Ok(cfg)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Config::default()),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

/// Resolve one setting, and say where the answer came from.
///
/// A flag beats the environment, the environment beats the file, and the file
/// beats the built-in default. `folio config` prints the source because a
/// surprising endpoint is usually an environment variable someone forgot.
pub(crate) fn resolve(
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

fn extract_host(authority: &str) -> &str {
    if authority.starts_with('[')
        && let Some(end) = authority.find(']')
    {
        return &authority[..=end];
    }
    authority.split_once(':').map_or(authority, |(h, _)| h)
}

fn endpoint_host(endpoint: &str) -> &str {
    let without_scheme = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .unwrap_or(endpoint);
    let authority = without_scheme.split(['/', '?']).next().unwrap_or_default();
    extract_host(authority)
}

/// Resolve the API key from environment or user config.
/// An API key never lives in project config. OPENAI_API_KEY is read only
/// when the endpoint host is api.openai.com.
pub(crate) fn resolve_api_key(
    endpoint: &str,
    user: Option<&str>,
) -> (Option<String>, &'static str) {
    if let Some(v) = std::env::var("FOLIO_API_KEY").ok().filter(|v| !v.is_empty()) {
        return (Some(v), "environment");
    }
    if endpoint_host(endpoint).eq_ignore_ascii_case("api.openai.com")
        && let Some(v) = std::env::var("OPENAI_API_KEY").ok().filter(|v| !v.is_empty())
    {
        return (Some(v), "environment");
    }
    if let Some(v) = user {
        return (Some(v.to_string()), "user config");
    }
    (None, "none")
}

pub(crate) fn resolve_allow_insecure(flag: bool, user: Option<bool>) -> bool {
    if flag {
        return true;
    }
    if let Ok(v) = std::env::var("FOLIO_ALLOW_INSECURE") {
        let v = v.trim();
        if v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes") {
            return true;
        }
        if v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no") {
            return false;
        }
    }
    user.unwrap_or(false)
}

fn is_loopback(host: &str) -> bool {
    let host = host.trim_matches('[').trim_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        return ip.is_loopback();
    }
    false
}

pub(crate) fn validate_transport(
    endpoint: &str,
    has_key: bool,
    allow_insecure: bool,
) -> Result<()> {
    if !has_key || allow_insecure {
        return Ok(());
    }
    if endpoint.starts_with("https://") {
        return Ok(());
    }
    if let Some(rest) = endpoint.strip_prefix("http://") {
        let authority = rest.split(['/', '?']).next().unwrap_or_default();
        let host = extract_host(authority);
        if is_loopback(host) {
            return Ok(());
        }
        bail!(
            "the endpoint at {endpoint} is unencrypted HTTP off the loopback; \
             sending credentials over plaintext HTTP is refused — use HTTPS, \
             or allow plaintext with --allow-insecure or `folio config set allow_insecure true`"
        );
    }
    bail!(
        "the endpoint at {endpoint} does not use HTTPS; \
         sending credentials over unencrypted transport is refused"
    );
}

/// The budget to re-index with, given what the index recorded.
///
/// An index written before folio recorded the budget has none, which reads back
/// as zero. Zero is not a budget: taken literally it truncates every section to
/// the empty string and embeds that, which is a silent and total corruption of
/// the index rather than an error. So it means "not recorded" and nothing else,
/// and `--max-chars` will not accept it.
pub(crate) fn budget(recorded: usize) -> usize {
    if recorded == 0 { MAX_CHARS } else { recorded }
}

pub(crate) fn at_least_one(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(0) => Err("a section budget of 0 characters would embed nothing".into()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_walk_stops_at_the_nearest_index_or_declaration() {
        let tmp = std::env::temp_dir().join(format!("folio-walk-{}", std::process::id()));
        let outer = tmp.join("outer");
        let inner = outer.join("inner");
        let deep = inner.join("a/b");
        fs::create_dir_all(&deep).expect("a tree to walk");
        fs::create_dir_all(outer.join(store::DIR)).expect("an index above");

        assert_eq!(corpus_root_from(&deep).as_deref(), Some(outer.as_path()));

        // A corpus that declares itself and has not been indexed yet still
        // ends the walk, so the refusal names it rather than an index above it.
        fs::write(inner.join(PROJECT_CONFIG), "").expect("a declaration between");
        assert_eq!(corpus_root_from(&deep).as_deref(), Some(inner.as_path()));

        // Outside any corpus there is nothing to find, and the caller keeps the
        // working directory they had.
        assert_eq!(corpus_root_from(&tmp), None);

        fs::remove_dir_all(&tmp).ok();
    }
    #[test]
    fn a_declaration_is_found_above_and_an_index_is_not() {
        let tmp = std::env::temp_dir().join(format!("folio-declaration-{}", std::process::id()));
        let sub = tmp.join("corpus/sub/deep");
        fs::create_dir_all(&sub).expect("a tree");
        // The walk canonicalizes, and the platform's temporary directory is a
        // symlink on macOS, so the fixture has to name what the walk will name.
        let tmp = tmp.canonicalize().expect("a real path");
        let sub = sub.canonicalize().expect("a real path");
        assert_eq!(nearest_declaration(&sub), None, "nothing above declares anything");

        let corpus = tmp.join("corpus");
        fs::write(corpus.join(PROJECT_CONFIG), "model: parent\n").expect("a declaration");
        assert_eq!(nearest_declaration(&sub), Some(corpus.join(PROJECT_CONFIG)));

        // The nearest one wins, so inheriting is a subtree looking up rather
        // than a corpus reaching down.
        let mid = tmp.join("corpus/sub");
        fs::write(mid.join(PROJECT_CONFIG), "model: child\n").expect("a nearer declaration");
        assert_eq!(nearest_declaration(&sub), Some(mid.join(PROJECT_CONFIG)));

        // An index above is not a declaration, and is reported as what it is.
        assert_eq!(enclosing_index(&sub), None);
        fs::create_dir_all(corpus.join(store::DIR)).expect("an index above");
        assert_eq!(enclosing_index(&sub).as_deref(), Some(corpus.as_path()));
        // An index at the root being indexed is the one about to be updated.
        fs::create_dir_all(sub.join(store::DIR)).expect("an index here");
        assert_eq!(enclosing_index(&sub).as_deref(), Some(corpus.as_path()));

        fs::remove_dir_all(&tmp).ok();
    }
    #[test]
    fn project_config_refuses_api_key() {
        let tmp = std::env::temp_dir().join(format!("folio-key-refuse-{}", std::process::id()));
        fs::create_dir_all(&tmp).expect("create dir");
        let proj_cfg = tmp.join(PROJECT_CONFIG);
        fs::write(&proj_cfg, "api_key: secret-should-refuse\n").expect("write folio.yaml");

        let err = read_config_at(&proj_cfg).unwrap_err();
        assert!(
            err.to_string().contains(
                "folio.yaml is committed with the corpus and must not contain credentials"
            )
        );

        // User config with api_key does not refuse
        let user_cfg = tmp.join("user_config.yaml");
        fs::write(&user_cfg, "api_key: secret-in-user\n").expect("write user config");
        let parsed = read_config_at(&user_cfg).expect("user config allows api_key");
        assert_eq!(parsed.api_key.as_deref(), Some("secret-in-user"));

        fs::remove_dir_all(&tmp).ok();
    }
    #[test]
    fn resolve_api_key_precedence() {
        unsafe {
            std::env::remove_var("FOLIO_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
        }

        // None configured
        let (k, src) = resolve_api_key("https://api.openai.com/v1", None);
        assert_eq!(k, None);
        assert_eq!(src, "none");

        // User config
        let (k, src) = resolve_api_key("https://api.openai.com/v1", Some("user-secret"));
        assert_eq!(k.as_deref(), Some("user-secret"));
        assert_eq!(src, "user config");

        // OPENAI_API_KEY beats user config only for api.openai.com
        unsafe { std::env::set_var("OPENAI_API_KEY", "openai-secret") };
        let (k, src) = resolve_api_key("https://api.openai.com/v1", Some("user-secret"));
        assert_eq!(k.as_deref(), Some("openai-secret"));
        assert_eq!(src, "environment");

        // OPENAI_API_KEY ignored for non-openai endpoints
        let (k, src) = resolve_api_key("http://127.0.0.1:8080/v1", Some("user-secret"));
        assert_eq!(k.as_deref(), Some("user-secret"));
        assert_eq!(src, "user config");

        // FOLIO_API_KEY beats OPENAI_API_KEY everywhere
        unsafe { std::env::set_var("FOLIO_API_KEY", "folio-secret") };
        let (k, src) = resolve_api_key("https://api.openai.com/v1", Some("user-secret"));
        assert_eq!(k.as_deref(), Some("folio-secret"));
        assert_eq!(src, "environment");

        let (k, src) = resolve_api_key("http://127.0.0.1:8080/v1", Some("user-secret"));
        assert_eq!(k.as_deref(), Some("folio-secret"));
        assert_eq!(src, "environment");

        unsafe {
            std::env::remove_var("FOLIO_API_KEY");
            std::env::remove_var("OPENAI_API_KEY");
        }
    }
    #[test]
    fn loopback_and_transport_validation() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("localhost"));
        assert!(is_loopback("::1"));
        assert!(is_loopback("[::1]"));
        assert!(is_loopback("127.0.1.1"));
        assert!(!is_loopback("192.0.2.1"));
        assert!(!is_loopback("api.openai.com"));
        assert!(!is_loopback("192.168.1.1"));

        // Without key: plaintext remote passes
        assert!(validate_transport("http://192.0.2.1:8080/v1/embeddings", false, false).is_ok());

        // With key: HTTPS passes
        assert!(validate_transport("https://api.openai.com/v1/embeddings", true, false).is_ok());

        // With key: loopback HTTP passes (IPv4, localhost, IPv6 with or without port)
        assert!(validate_transport("http://127.0.0.1:8080/v1/embeddings", true, false).is_ok());
        assert!(validate_transport("http://localhost:8080/v1/embeddings", true, false).is_ok());
        assert!(validate_transport("http://[::1]/v1/embeddings", true, false).is_ok());
        assert!(validate_transport("http://[::1]:8080/v1/embeddings", true, false).is_ok());

        // With key: non-loopback plaintext HTTP fails
        let err =
            validate_transport("http://192.0.2.1:8080/v1/embeddings", true, false).unwrap_err();
        assert!(err.to_string().contains("sending credentials over plaintext HTTP is refused"));

        // With key: non-loopback plaintext HTTP passes when allow_insecure is true
        assert!(validate_transport("http://192.0.2.1:8080/v1/embeddings", true, true).is_ok());

        // Non-http/https scheme is refused when key is present
        assert!(validate_transport("ftp://example.com/v1/embeddings", true, false).is_err());

        // FOLIO_ALLOW_INSECURE can enable or disable
        unsafe { std::env::set_var("FOLIO_ALLOW_INSECURE", "1") };
        assert!(resolve_allow_insecure(false, Some(false)));
        unsafe { std::env::set_var("FOLIO_ALLOW_INSECURE", "0") };
        assert!(!resolve_allow_insecure(false, Some(true)));
        unsafe { std::env::set_var("FOLIO_ALLOW_INSECURE", "false") };
        assert!(!resolve_allow_insecure(false, Some(true)));
        unsafe { std::env::remove_var("FOLIO_ALLOW_INSECURE") };
    }
}

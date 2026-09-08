//! The service file that runs an embeddings server, and the flags folio cannot
//! do without.
//!
//! folio manages no server. It prints a unit for whatever manages services on
//! the machine, and it knows the two servers it prints well enough to refuse a
//! flag belonging to the other one — D-01M1XRDKA9FJ1V. Installing it is the
//! caller's.

use anyhow::{Context, Result, bail};
use std::path::Path;

use crate::config::{
    DEFAULT_ENDPOINT, PROJECT_CONFIG, config_path, nearest_declaration, read_config_at, resolve,
};

/// A server `folio unit` knows how to start.
///
/// folio runs no model and learns nothing about one from this; what it holds
/// per backend is a recipe, and D-01M1XRDKA9FJ1V says why holding one beats
/// handing the caller a command to write.
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum Backend {
    #[value(name = "llama.cpp")]
    LlamaCpp,
    #[value(name = "tei")]
    Tei,
}

/// The host and port a service should bind, read from the endpoint folio uses.
fn host_port(endpoint: &str) -> Result<(String, u16)> {
    let rest = endpoint.split_once("://").map_or(endpoint, |(_, r)| r);
    let authority = rest.split(['/', '?']).next().unwrap_or_default();
    let (host, port) = authority
        .rsplit_once(':')
        .with_context(|| format!("{endpoint} names no port, so a service cannot bind it"))?;
    let port: u16 = port.parse().with_context(|| format!("{port} is not a port number"))?;
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
            "-hf".into(),
            hf.into(),
            "--hf-file".into(),
            hf_file.into(),
            "--pooling".into(),
            pooling.into(),
            "-c".into(),
            context.to_string(),
            "-b".into(),
            context.to_string(),
            "-ub".into(),
            context.to_string(),
        ],
        Backend::Tei => vec![
            "--model-id".into(),
            hf.into(),
            "--auto-truncate".into(),
            "false".into(),
            "--max-batch-tokens".into(),
            context.to_string(),
        ],
    };
    args.extend(["--host".to_string(), host, "--port".to_string(), port.to_string()]);
    args
}

// Eight flags of a service file, each printed once and read nowhere else.
#[expect(clippy::too_many_arguments)]
pub(crate) fn cmd_unit(
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
        Backend::LlamaCpp => {
            ("llama-server", hf.unwrap_or("keisuke-miyako/gte-modernbert-base-gguf"))
        }
        Backend::Tei => ("text-embeddings-router", hf.unwrap_or("Alibaba-NLP/gte-modernbert-base")),
    };
    let hf_file = hf_file.unwrap_or("gte-modernbert-base-Q8_0.gguf");
    let pooling = pooling.unwrap_or("cls");
    let declared = nearest_declaration(root).unwrap_or_else(|| root.join(PROJECT_CONFIG));
    let proj = read_config_at(&declared)?;
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
    eprintln!(
        "{name} is not on PATH; the service file names it unqualified, and a service manager will not find it"
    );
    name.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let args = backend_args(
            Backend::Tei,
            "org/model",
            "ignored",
            "ignored",
            8192,
            "127.0.0.1".into(),
            8080,
        );
        let pairs: Vec<(&str, &str)> =
            args.windows(2).map(|w| (w[0].as_str(), w[1].as_str())).collect();
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
        let args = backend_args(
            Backend::LlamaCpp,
            "repo",
            "file.gguf",
            "cls",
            8192,
            "127.0.0.1".into(),
            8080,
        );
        assert_eq!(
            args,
            [
                "--embeddings",
                "-hf",
                "repo",
                "--hf-file",
                "file.gguf",
                "--pooling",
                "cls",
                "-c",
                "8192",
                "-b",
                "8192",
                "-ub",
                "8192",
                "--host",
                "127.0.0.1",
                "--port",
                "8080"
            ]
        );
    }
}

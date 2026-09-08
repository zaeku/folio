//! Talking to the endpoint, and the arithmetic on what comes back.
//!
//! folio runs no model. It sends text to an OpenAI-compatible service and
//! compares what returns, which is why a vector space is what an endpoint
//! answers rather than what it is called — D-01M1PP6HKHF97G. Vectors arrive
//! L2-normalized here so that ranking is a plain dot product.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::validate_transport;

const BATCH: usize = 32;

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
pub(crate) fn calibrate(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    allow_insecure: bool,
    longest: &str,
    budget: usize,
) -> Result<usize> {
    /// Below this, a section is too short to carry a section's meaning, and a
    /// server refusing it is refusing for some other reason.
    const FLOOR: usize = 256;

    let mut budget = budget;
    loop {
        let probe: String = longest.chars().take(budget).collect();
        match embed(endpoint, model, api_key, allow_insecure, &[probe]) {
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
pub(crate) enum EmbedError {
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
pub(crate) fn embed(
    endpoint: &str,
    model: &str,
    api_key: Option<&str>,
    allow_insecure: bool,
    inputs: &[String],
) -> Result<Vec<Vec<f32>>, EmbedError> {
    if let Err(e) = validate_transport(endpoint, api_key.is_some(), allow_insecure) {
        return Err(EmbedError::NoAnswer(e));
    }
    let mut out: Vec<Vec<f32>> = vec![Vec::new(); inputs.len()];
    for (bi, batch) in inputs.chunks(BATCH).enumerate() {
        // A server that answered is not a server that is absent, and only one
        // of those is worth retrying. Status is handled here rather than raised
        // as a transport error so that the server's own sentence survives: it
        // is the sentence that says which input was too long for its batch.
        let mut req = ureq::post(endpoint);
        if let Some(key) = api_key {
            req = req.header("Authorization", &format!("Bearer {key}"));
        }
        let mut response = req
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

pub(crate) fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in &mut v {
            *x /= n;
        }
    }
    v
}

pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// How close the endpoint answering now is to the one that built the index.
///
/// A different length is a different space and not a near miss, so it scores 0
/// rather than comparing the dimensions they happen to share.
pub(crate) fn same_space(now: &[f32], then: &[f32]) -> f32 {
    if now.len() != then.len() || then.is_empty() {
        return 0.0;
    }
    dot(now, then)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_different_dimension_is_a_different_space_and_not_a_near_miss() {
        let then = vec![1.0, 0.0, 0.0];
        assert_eq!(same_space(&[1.0, 0.0, 0.0], &then), 1.0);
        assert_eq!(
            same_space(&[1.0, 0.0], &then),
            0.0,
            "a shorter vector must not score on the dimensions it shares"
        );
        assert_eq!(
            same_space(&[1.0, 0.0, 0.0], &[]),
            0.0,
            "an index with no fingerprint is not a match"
        );
    }
}

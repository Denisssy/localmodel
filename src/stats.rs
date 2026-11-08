#[derive(Default, Clone)]
pub struct Stats {
    pub alive: bool,
    pub ctx: u64,
    pub last_pre: Option<f64>,
    pub last_dec: Option<f64>,
    pub dec_tokens: f64,
    pub dec_secs: f64,
    pub pre_tokens: f64,
    pub pre_secs: f64,
    pub reqs: u64,
}

fn num_before(s: &str, needle: &str) -> Option<f64> {
    let idx = s.find(needle)?;
    let tok = s[..idx].split_whitespace().last()?;
    tok.trim_matches(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .parse()
        .ok()
}

fn tokens_after_ms(s: &str) -> Option<f64> {
    let idx = s.find("ms /")?;
    s[idx + 4..].split_whitespace().next()?.parse().ok()
}

impl Stats {
    pub fn avg_dec(&self) -> Option<f64> {
        (self.dec_secs > 0.0).then(|| self.dec_tokens / self.dec_secs)
    }
    pub fn observe(&mut self, line: &str) {
        if line.contains("listening on http") || line.contains("model loaded") {
            self.alive = true;
        }
        if !line.contains("tokens per second") {
            return;
        }
        let is_prompt = line.contains("prompt eval");
        let is_eval = line.contains("eval time");
        if !is_prompt && !is_eval {
            return;
        }
        let (Some(tps), Some(ms), Some(toks)) = (
            num_before(line, "tokens per second"),
            num_before(line, " ms"),
            tokens_after_ms(line),
        ) else {
            return;
        };
        let secs = ms / 1000.0;
        if is_prompt {
            self.pre_tokens += toks;
            self.pre_secs += secs;
            self.last_pre = Some(tps);
        } else {
            self.dec_tokens += toks;
            self.dec_secs += secs;
            self.last_dec = Some(tps);
            self.reqs += 1;
        }
    }
}

/// One scrape of a Prometheus `/metrics` endpoint.
///
/// Both llama.cpp and Strata expose the same counter names for tokens; Strata
/// adds its own around them. We match on the suffix so a prefix change does not
/// break the dashboard.
#[derive(Default, Clone, Copy)]
pub struct MetricSample {
    pub predicted: f64,
    pub predicted_secs: f64,
    pub prompt: f64,
    pub prompt_secs: f64,
    pub requests: f64,
}

fn metric_value(line: &str) -> Option<(&str, f64)> {
    if line.starts_with('#') || line.trim().is_empty() {
        return None;
    }
    let name = line.split(['{', ' ']).next()?.trim();
    let value = line.rsplit(' ').next()?.trim().parse::<f64>().ok()?;
    Some((name, value))
}

pub fn parse_metrics(text: &str) -> Option<MetricSample> {
    let mut s = MetricSample::default();
    let mut seen = false;
    for line in text.lines() {
        let Some((name, v)) = metric_value(line) else { continue };
        let n = name.to_lowercase();
        if n.ends_with("tokens_predicted_total") {
            s.predicted = v;
            seen = true;
        } else if n.ends_with("tokens_predicted_seconds_total") {
            s.predicted_secs = v;
        } else if n.ends_with("prompt_tokens_total") {
            s.prompt = v;
            seen = true;
        } else if n.ends_with("prompt_tokens_seconds_total") {
            s.prompt_secs = v;
        } else if n.ends_with("requests_total") || n.ends_with("requests_processed_total") {
            s.requests = v;
        }
    }
    seen.then_some(s)
}

/// Strata's own `/metrics` is a JSON snapshot, not Prometheus text: a rolling
/// `requests` array (each with its own timings) plus a `live` block for the
/// request in flight. llama.cpp keeps the Prometheus format.
fn f(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0)
}

pub fn parse_strata_metrics(text: &str) -> Option<MetricSample> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let reqs = v.get("requests")?.as_array()?;
    let mut s = MetricSample { requests: reqs.len() as f64, ..Default::default() };
    for r in reqs {
        s.predicted += f(r, "engine_generated");
        s.predicted_secs += f(r, "decode_ms") / 1000.0;
        s.prompt += f(r, "prompt_read");
        s.prompt_secs += f(r, "prompt_ms") / 1000.0;
    }
    Some(s)
}

/// Turns cumulative counters into the per-second numbers the dashboard shows.
pub struct MetricsPoller {
    prev: Option<MetricSample>,
    pub stats: Stats,
}

impl MetricsPoller {
    pub fn new(ctx: u64) -> Self {
        Self {
            prev: None,
            stats: Stats { ctx, ..Default::default() },
        }
    }

    pub fn observe(&mut self, text: &str) -> bool {
        if text.trim_start().starts_with('{') {
            return self.observe_strata(text);
        }
        let Some(cur) = parse_metrics(text) else {
            return false;
        };
        self.stats.alive = true;
        if let Some(prev) = self.prev {
            let dp = cur.predicted - prev.predicted;
            let ds = cur.predicted_secs - prev.predicted_secs;
            if dp > 0.0 && ds > 0.0 {
                self.stats.last_dec = Some(dp / ds);
            }
            let pp = cur.prompt - prev.prompt;
            let ps = cur.prompt_secs - prev.prompt_secs;
            if pp > 0.0 && ps > 0.0 {
                self.stats.last_pre = Some(pp / ps);
            }
            if cur.requests > prev.requests {
                self.stats.reqs = cur.requests as u64;
            }
        }
        if cur.predicted_secs > 0.0 {
            self.stats.dec_tokens = cur.predicted;
            self.stats.dec_secs = cur.predicted_secs;
        }
        if cur.prompt_secs > 0.0 {
            self.stats.pre_tokens = cur.prompt;
            self.stats.pre_secs = cur.prompt_secs;
        }
        self.prev = Some(cur);
        true
    }

    /// Strata: the snapshot already carries per-request rates, so take them
    /// directly instead of diffing counters.
    fn observe_strata(&mut self, text: &str) -> bool {
        let Some(sample) = parse_strata_metrics(text) else {
            return false;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(text) else {
            return false;
        };
        let Some(reqs) = v.get("requests").and_then(|r| r.as_array()) else {
            return false;
        };
        self.stats.alive = true;
        self.stats.reqs = self.stats.reqs.max(sample.requests as u64);
        self.stats.dec_tokens = sample.predicted;
        self.stats.dec_secs = sample.predicted_secs;
        self.stats.pre_tokens = sample.prompt;
        self.stats.pre_secs = sample.prompt_secs;
        if let Some(last) = reqs.last() {
            if let Some(x) = last.get("decode_tok_s").and_then(|x| x.as_f64()) {
                self.stats.last_dec = Some(x);
            }
            let pms = f(last, "prompt_ms");
            if pms > 0.0 {
                self.stats.last_pre = Some(f(last, "prompt_read") / (pms / 1000.0));
            }
        }
        if let Some(live) = v.get("live") {
            if let Some(x) = live.get("tok_s").and_then(|x| x.as_f64()) {
                self.stats.last_dec = Some(x);
            }
            if let Some(x) = live.get("prefill_tok_s_mean").and_then(|x| x.as_f64()) {
                self.stats.last_pre = Some(x);
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strata_json_snapshot_is_parsed() {
        let text = r#"{"engine":{"model":"m","max_context":131072},
            "live":{"state":"idle","tok_s":null,"prefill_tok_s_mean":null},
            "requests":[
              {"engine_generated":8,"decode_ms":154.8,"prompt_read":16,"prompt_ms":168.9,"decode_tok_s":51.7}
            ]}"#;
        let s = parse_strata_metrics(text).expect("json");
        assert_eq!(s.requests, 1.0);
        assert_eq!(s.predicted, 8.0);
        assert!((s.predicted_secs - 0.1548).abs() < 1e-6);
        assert_eq!(s.prompt, 16.0);

        let mut p = MetricsPoller::new(131072);
        assert!(p.observe(text));
        assert_eq!(p.stats.last_dec, Some(51.7));
        assert!(p.stats.last_pre.unwrap() > 90.0);
    }
}

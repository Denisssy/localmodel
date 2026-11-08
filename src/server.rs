use std::fs;
use std::net::TcpListener;
use std::process::Command;

pub fn free_port(start: u16) -> u16 {
    for p in start..start.saturating_add(100) {
        if TcpListener::bind(("127.0.0.1", p)).is_ok() {
            return p;
        }
    }
    start
}

pub struct GpuInfo {
    pub name: String,
    pub util: f64,
    pub mem_used: f64,
    pub mem_total: f64,
    pub temp: f64,
    pub power_draw: f64,
    pub power_limit: f64,
}

pub fn gpu_info() -> Option<GpuInfo> {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s.lines().next()?;
    let p: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
    if p.len() < 7 {
        return None;
    }
    Some(GpuInfo {
        name: p[0].to_string(),
        util: p[1].parse().unwrap_or(0.0),
        mem_used: p[2].parse().unwrap_or(0.0),
        mem_total: p[3].parse().unwrap_or(0.0),
        temp: p[4].parse().unwrap_or(0.0),
        power_draw: p[5].parse().unwrap_or(0.0),
        power_limit: p[6].parse().unwrap_or(0.0),
    })
}

pub fn gpu_util() -> Option<f64> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    s.lines().next()?.trim().parse::<f64>().ok()
}

pub fn gpu_mem() -> Option<(f64, f64)> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    let mut it = s.lines().next()?.split(',');
    let used: f64 = it.next()?.trim().parse().ok()?;
    let total: f64 = it.next()?.trim().parse().ok()?;
    Some((used, total))
}

/// Machine-wide RAM: `used = MemTotal - MemAvailable`, i.e. what tools like
/// `free` and desktop monitors call "used". Page cache is *not* included.
pub struct SysMem {
    pub total_kb: u64,
    pub avail_kb: u64,
}

pub fn sys_mem() -> Option<SysMem> {
    let s = fs::read_to_string("/proc/meminfo").ok()?;
    let get = |key: &str| -> Option<u64> {
        s.lines()
            .find_map(|l| l.strip_prefix(key))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
    };
    Some(SysMem {
        total_kb: get("MemTotal:")?,
        avail_kb: get("MemAvailable:")?,
    })
}

/// Resident vs anonymous memory of one process. `rss` includes the mmap'ed
/// GGUF, which lives in the page cache and is reclaimable — that is why RSS
/// can look bigger than the RAM the whole system reports as used.
pub struct ProcMem {
    pub rss_kb: u64,
    pub anon_kb: u64,
}

pub fn proc_mem(pid: u32) -> Option<ProcMem> {
    if let Ok(s) = fs::read_to_string(format!("/proc/{pid}/smaps_rollup")) {
        let get = |key: &str| -> u64 {
            s.lines()
                .find_map(|l| l.strip_prefix(key))
                .and_then(|r| r.split_whitespace().next())
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        let rss = get("Rss:");
        if rss > 0 {
            return Some(ProcMem {
                rss_kb: rss,
                anon_kb: get("Private_Dirty:"),
            });
        }
    }
    rss_kb(pid).map(|k| ProcMem { rss_kb: k, anon_kb: 0 })
}

pub fn rss_kb(pid: u32) -> Option<u64> {
    let s = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    for l in s.lines() {
        if let Some(r) = l.strip_prefix("VmRSS:") {
            return r.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

pub struct Server {
    pub pid: u32,
    pub cmd: String,
}

pub fn list_servers(models_dir: &str) -> Vec<Server> {
    let mut v = Vec::new();
    let Ok(rd) = fs::read_dir("/proc") else { return v };
    for e in rd.flatten() {
        let Ok(pid) = e.file_name().to_string_lossy().parse::<u32>() else { continue };
        let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) else { continue };
        if raw.is_empty() {
            continue;
        }
        let cmd = raw
            .split(|b| *b == 0)
            .map(|c| String::from_utf8_lossy(c).to_string())
            .collect::<Vec<_>>()
            .join(" ");
        if cmd.contains("llama-server") && cmd.contains(models_dir) {
            v.push(Server { pid, cmd: cmd.trim().to_string() });
        }
    }
    v
}

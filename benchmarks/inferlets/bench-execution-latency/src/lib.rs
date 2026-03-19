use inferlet::wstd::time::Instant;
use inferlet::{Args, Result, anyhow};

const HELP: &str = "\
Usage: bench-execution-latency [OPTIONS]

An inferlet for benchmarking API execution latency.

Options:
  -i, --index <INTEGER>          Optional sample index key.
  -l, --layer <STRING>           Layer to benchmark: control|inference [default: inference]
  -a, --aggregate-size <UINT>    Aggregate and report stored samples.
  -h, --help                     Print help information.";

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(String::new());
    }

    let index: Option<u32> = args.value_from_str(["-i", "--index"]).ok();
    let layer: String = args
        .value_from_str(["-l", "--layer"])
        .unwrap_or_else(|_| "inference".to_string());
    let aggregate_size: Option<u32> = args.value_from_str(["-a", "--aggregate-size"]).ok();

    if let Some(size) = aggregate_size {
        let mut latencies: Vec<u128> = Vec::with_capacity(size as usize);

        for i in 0..size {
            let key = format!("latency-{}", i);
            if let Some(value_str) = inferlet::store_get(&key) {
                if let Ok(latency) = value_str.parse::<u128>() {
                    latencies.push(latency);
                }
            }
        }

        if latencies.is_empty() {
            return Ok("{\"samples\":0,\"mean\":0.0,\"median\":0.0,\"std_dev\":0.0}".to_string());
        }

        let count = latencies.len();
        let sum: u128 = latencies.iter().sum();
        let mean = sum as f64 / count as f64;

        latencies.sort_unstable();
        let median = if count % 2 == 0 {
            let mid = count / 2;
            (latencies[mid - 1] + latencies[mid]) as f64 / 2.0
        } else {
            latencies[count / 2] as f64
        };

        let variance = latencies
            .iter()
            .map(|value| {
                let diff = *value as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / count as f64;
        let std_dev = variance.sqrt();

        let payload = format!(
            "{{\"samples\":{},\"mean\":{:.2},\"median\":{:.2},\"std_dev\":{:.2}}}",
            count, mean, median, std_dev
        );
        return Ok(payload);
    }

    let start_time = Instant::now();

    match layer.as_str() {
        "control" => {
            let _ = inferlet::debug_query("ping").await;
        }
        "inference" => {
            let model = inferlet::get_auto_model();
            let queue = model.create_queue();
            let _ = queue.debug_query("ping").await;
        }
        _ => {
            return Err(anyhow!("Invalid layer: {}", layer));
        }
    }

    let elapsed = start_time.elapsed();
    let key = format!("latency-{}", index.unwrap_or(0));
    inferlet::store_set(&key, &elapsed.as_micros().to_string());

    Ok(String::new())
}

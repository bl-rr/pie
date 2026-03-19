use inferlet::{Args, Result};

const HELP: &str = "\
Usage: bench-spawn-time [OPTIONS]

An inferlet for benchmarking spawn latency.

Options:
  -m, --message <STRING>   The message to print.
                           [default: Hello from the inferlet!]
  -h, --help               Print help information.";

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    let msg: String = args
        .value_from_str(["-m", "--message"])
        .unwrap_or_else(|_| "Hello from the inferlet!".to_string());

    println!("{}", msg);
    Ok(())
}

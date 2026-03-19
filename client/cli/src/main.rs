use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use pie_client::client::Client;
use pie_client::client::InstanceEvent;
use pie_client::message::EventCode;
use serde::Deserialize;

#[derive(Debug, Parser)]
#[command(name = "pie-submit-eval")]
#[command(about = "Submit and run inferlets against an eval Pie server")]
struct Cli {
    /// Pie server host
    #[arg(long)]
    host: Option<String>,

    /// Pie server port
    #[arg(long)]
    port: Option<u16>,

    /// Username used for authentication
    #[arg(long, default_value_t = whoami::username())]
    username: String,

    /// Explicit path to Pie config.toml
    #[arg(long, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Path to inferlet wasm
    wasm_path: PathBuf,

    /// Path to Pie manifest
    manifest_path: PathBuf,

    /// Arguments passed to inferlet after '--'
    #[arg(trailing_var_arg = true)]
    inferlet_args: Vec<String>,
}

#[derive(Debug, Default, Deserialize)]
struct PieConfig {
    host: Option<String>,
    port: Option<u16>,
}

fn default_eval_config_path() -> PathBuf {
    if let Ok(path) = env::var("PIE_CONFIG_PATH") {
        return PathBuf::from(path);
    }

    let pie_home = env::var("PIE_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| dirs::home_dir().map(|home| home.join(".pie-eval")))
        .unwrap_or_else(|| PathBuf::from(".pie-eval"));

    pie_home.join("config.toml")
}

fn load_config(path: &Path) -> Result<PieConfig> {
    if !path.exists() {
        return Ok(PieConfig::default());
    }

    let content =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    toml::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut cli = Cli::parse();
    if !cli.inferlet_args.is_empty() && cli.inferlet_args[0] == "--" {
        cli.inferlet_args.remove(0);
    }

    let config_path = cli.config.unwrap_or_else(default_eval_config_path);
    let config = load_config(&config_path)?;
    let host = cli
        .host
        .or(config.host)
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = cli.port.or(config.port).unwrap_or(10009);

    let uri = format!("ws://{}:{}", host, port);
    let client = Client::connect(&uri)
        .await
        .with_context(|| format!("failed to connect to {}", uri))?;

    client
        .authenticate(&cli.username, &None)
        .await
        .with_context(|| format!("authentication failed for user '{}'", cli.username))?;

    client
        .install_program(&cli.wasm_path, &cli.manifest_path)
        .await?;

    // get name from manifest
    let manifest = fs::read_to_string(&cli.manifest_path)?;
    let manifest: toml::Value = toml::from_str(&manifest)?;
    let name = manifest
        .get("package")
        .and_then(|pkg| pkg.get("name"))
        .and_then(|name| name.as_str())
        .unwrap();

    let version = manifest
        .get("package")
        .and_then(|pkg| pkg.get("version"))
        .and_then(|version| version.as_str())
        .unwrap();

    // Launch
    let mut instance = client
        .launch_instance(
            format!("{}@{}", name, version),
            cli.inferlet_args.clone(),
            false,
        )
        .await?;

    // Receive output
    loop {
        match instance.recv().await? {
            InstanceEvent::Stdout(text) => print!("{}", text),
            InstanceEvent::Stderr(text) => eprint!("{}", text),
            InstanceEvent::Event {
                code: EventCode::Completed,
                message,
            } => {
                if !message.is_empty() {
                    println!("{}", message);
                }
                break;
            }
            InstanceEvent::Event {
                code: EventCode::Exception,
                message,
            } => {
                eprintln!("Error: {}", message);
                std::process::exit(1);
            }
            InstanceEvent::Event {
                code: EventCode::Aborted,
                message,
            } => {
                eprintln!("Error: {}", message);
                std::process::exit(1);
            }
            InstanceEvent::Event {
                code: EventCode::ServerError,
                message,
            } => {
                eprintln!("Error: {}", message);
                std::process::exit(1);
            }
            InstanceEvent::Event {
                code: EventCode::OutOfResources,
                message,
            } => {
                eprintln!("Error: {}", message);
                std::process::exit(1);
            }
            _ => {}
        }
    }

    Ok(())
}

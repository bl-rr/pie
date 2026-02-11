use std::fs;
use std::path::Path;

use anyhow::Result;
use pie_client::client::Client;
use pie_client::client::InstanceEvent;
use pie_client::client::hash_blob;
use pie_client::message::EventCode;

#[tokio::main]
async fn main() -> Result<()> {
    // Connect to server
    let client = Client::connect("ws://127.0.0.1:9099").await?;

    // Authenticate
    client.authenticate("username", &None).await?;

    // read in the name of the wasm as argument
    let args = std::env::args().collect::<Vec<String>>();
    let wasm_path = Path::new(&args[1]);

    let manifest_path = Path::new(&args[2]);
    client.install_program(wasm_path, manifest_path).await?;

    // get name from manifest
    let manifest = fs::read_to_string(manifest_path)?;
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

    // Get arguments
    let args = std::env::args().collect::<Vec<String>>();
    let args = args.iter().skip(2).cloned().collect::<Vec<String>>();

    // Launch
    let mut instance = client
        .launch_instance(format!("{}@{}", name, version), args, false)
        .await?;

    // Receive output
    loop {
        match instance.recv().await? {
            InstanceEvent::Stdout(text) => print!("{}", text),
            // event code completed and any message
            InstanceEvent::Event {
                code: EventCode::Completed,
                message: msg,
            } => {
                println!("{}", msg);
                println!("Inferlet {} completed", instance.id());
                break;
            }
            InstanceEvent::Event {
                code: EventCode::Exception,
                message: msg,
            } => {
                eprintln!("Error: {}", msg);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}

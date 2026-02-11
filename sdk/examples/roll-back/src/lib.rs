//! Roll-back efficiency benchmark for Pie's KV-cache page-level manipulation.
//!
//! This example generates tokens in bursts of `roll_back_period`, rolls back
//! half of them (`roll_back_period / 2`), and continues until `max_tokens`
//! net tokens have been generated. At each rollback point the generated text
//! is snapshotted so correctness can be verified.

use inferlet::{Args, Result, Sampler};
use std::time::Instant;

#[inferlet::main]
async fn main(mut args: Args) -> Result<()> {
    let prompt: String = args
        .value_from_str(["-p", "--prompt"])
        .unwrap_or("Explain the Rust borrow checker in detail.".to_string());
    let max_tokens: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(128);
    let roll_back_period: usize = args
        .value_from_str(["-r", "--roll-back-period"])
        .unwrap_or(128);
    let temperature: f32 = args.value_from_str(["-t", "--temperature"]).unwrap_or(0.0);

    // The number of tokens we actually keep after each rollback.
    let keep = roll_back_period - roll_back_period / 2;

    let model = inferlet::get_auto_model();
    let tokenizer = model.get_tokenizer();
    let sampler = if temperature > 0.0 {
        Sampler::Multinomial { temperature }
    } else {
        Sampler::greedy()
    };

    let mut ctx = model.create_context();
    ctx.fill_system("You are a helpful, respectful and honest assistant.");
    ctx.fill_user(&prompt);

    println!(
        "Starting roll-back benchmark: max_tokens={}, roll_back_period={}, keep={}",
        max_tokens, roll_back_period, keep
    );

    let start = Instant::now();

    // All generated token ids (net — after rollbacks).
    let mut generated_token_ids: Vec<u32> = Vec::new();
    // Tokens generated in the current burst (before rollback).
    let mut burst_token_ids: Vec<u32> = Vec::new();
    // Snapshots of text at each rollback point.
    let mut snapshots: Vec<String> = Vec::new();

    let mut total_generated: usize = 0; // gross tokens generated (includes rolled-back)
    let mut total_rolled_back: usize = 0;
    let mut num_rollbacks: usize = 0;

    while generated_token_ids.len() < max_tokens {
        // --- Generate one token ---
        let next_token_id = ctx.decode_step(&sampler).await;
        ctx.fill_token(next_token_id);

        burst_token_ids.push(next_token_id);
        total_generated += 1;

        // --- End of burst: time to roll back ---
        if burst_token_ids.len() == roll_back_period {
            // Snapshot the full text *before* rolling back.
            let snapshot_tokens: Vec<u32> = generated_token_ids
                .iter()
                .chain(burst_token_ids.iter())
                .copied()
                .collect();
            let snapshot_text = tokenizer.detokenize(&snapshot_tokens);
            snapshots.push(snapshot_text);

            let roll_back_amount = roll_back_period / 2;

            // Keep the first `keep` tokens of this burst.
            let kept: Vec<u32> = burst_token_ids[..keep].to_vec();
            generated_token_ids.extend(&kept);

            // --- Roll back the KV cache ---
            // We need to remove `roll_back_amount` tokens worth of KV state.
            // Also truncate the context's token_ids and position_ids to match.
            ctx.token_ids
                .truncate(ctx.token_ids.len() - roll_back_amount);
            ctx.position_ids
                .truncate(ctx.position_ids.len() - roll_back_amount);
            ctx.shrink_kv_pages(roll_back_amount);

            // The last kept token needs to be re-queued as pending so the
            // next decode_step has a seed token.
            let last_kept = ctx.token_ids.pop().unwrap();
            ctx.position_ids.pop();
            ctx.shrink_kv_pages(1);
            ctx.fill_token(last_kept);

            total_rolled_back += roll_back_amount;
            num_rollbacks += 1;
            burst_token_ids.clear();

            println!(
                "  Rollback #{}: net tokens so far = {}, rolled back {} tokens",
                num_rollbacks,
                generated_token_ids.len(),
                roll_back_amount,
            );
        }
    }

    // If there are leftover burst tokens, keep them (up to what's needed to reach max_tokens).
    if !burst_token_ids.is_empty() {
        let remaining = max_tokens - generated_token_ids.len();
        let to_keep = burst_token_ids.len().min(remaining);
        generated_token_ids.extend(&burst_token_ids[..to_keep]);
    }

    let elapsed = start.elapsed();
    let final_text = tokenizer.detokenize(&generated_token_ids);

    // --- Print results ---
    println!("\n=== Roll-back Benchmark Results ===");
    println!("Net tokens generated : {}", generated_token_ids.len());
    println!("Gross tokens generated: {}", total_generated);
    println!("Total rolled back    : {}", total_rolled_back);
    println!("Number of rollbacks  : {}", num_rollbacks);
    println!("Total elapsed        : {:?}", elapsed);

    if !generated_token_ids.is_empty() {
        println!(
            "Per-token latency (net): {:?}",
            elapsed / generated_token_ids.len() as u32
        );
    }
    if total_generated > 0 {
        println!(
            "Per-token latency (gross): {:?}",
            elapsed / total_generated as u32
        );
    }

    println!("\n=== Final Output ===\n{}\n", final_text);

    // Print snapshots for verification.
    println!("=== Snapshots at Rollback Points ===");
    for (i, snap) in snapshots.iter().enumerate() {
        println!("--- Snapshot #{} (before rollback) ---", i + 1);
        println!("{}\n", snap);
    }

    Ok(())
}

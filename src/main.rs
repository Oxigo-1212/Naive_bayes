use csv::{ReaderBuilder, StringRecord};
use spam_filter::train;
use std::error::Error;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(root).join("data/emails.csv");
    let model = train(&path)?;

    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let header = rdr.headers()?.clone();
    let header_len = header.len();

    // Load all records into memory so we can split them
    let all_records: Vec<StringRecord> = rdr.records().filter_map(|r| r.ok()).collect();
    let midpoint = all_records.len() / 2;
    let (_train_set, test_set) = all_records.split_at(midpoint);

    // --- Testing phase: classify the second half ---
    let mut correct = 0u32;
    let mut total_test = 0u32;
    let mut true_pos = 0u32;
    let mut false_pos = 0u32;
    let mut true_neg = 0u32;
    let mut false_neg = 0u32;

    for record in test_set {
        let actual_spam = record.get(header_len - 1) == Some("1");
        let actual_label = if actual_spam { "spam" } else { "ham" };

        // Extract this document's word counts
        let word_counts: Vec<u32> = (1..(header_len - 1))
            .map(|i| record[i].parse().unwrap_or(0))
            .collect();

        let (predicted, _words) = model.classify(&word_counts);

        if predicted == actual_label {
            correct += 1;
        }

        match (predicted, actual_label) {
            ("spam", "spam") => true_pos += 1,
            ("spam", "ham") => false_pos += 1,
            ("ham", "ham") => true_neg += 1,
            ("ham", "spam") => false_neg += 1,
            _ => {}
        }

        total_test += 1;
    }

    let accuracy = correct as f64 / total_test as f64 * 100.0;
    let precision = if true_pos + false_pos > 0 {
        true_pos as f64 / (true_pos + false_pos) as f64
    } else {
        0.0
    };
    let recall = if true_pos + false_neg > 0 {
        true_pos as f64 / (true_pos + false_neg) as f64
    } else {
        0.0
    };

    println!("\n--- Results on {} test emails ---", total_test);
    println!("Accuracy:  {:.2}%", accuracy);
    println!("Precision: {:.4}", precision);
    println!("Recall:    {:.4}", recall);
    println!(
        "Confusion: TP={} FP={} TN={} FN={}",
        true_pos, false_pos, true_neg, false_neg
    );

    Ok(())
}

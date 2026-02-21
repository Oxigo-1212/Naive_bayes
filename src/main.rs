use csv::{ReaderBuilder, StringRecord};
use std::error::Error;
use std::path::Path;

fn naive_bayes_classify<'a>(
    headers: &'a StringRecord,
    log_prior_spam: f64,
    log_prior_ham: f64,
    log_likelihood_spam: &[f64],
    log_likelihood_ham: &[f64],
    word_counts: &[u32],
) -> (&'static str, Vec<&'a str>) {
    let spam_score = log_prior_spam
        + word_counts
            .iter()
            .zip(log_likelihood_spam)
            .map(|(&count, &log_prob)| count as f64 * log_prob)
            .sum::<f64>();

    let ham_score = log_prior_ham
        + word_counts
            .iter()
            .zip(log_likelihood_ham)
            .map(|(&count, &log_prob)| count as f64 * log_prob)
            .sum::<f64>();

    let label = if spam_score > ham_score {
        "spam"
    } else {
        "ham"
    };

    let words = word_counts
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0)
        .map(|(i, _)| headers.get(i + 1).unwrap_or(""))
        .collect();

    (label, words)
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = Path::new(root).join("data/emails.csv");
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let header = rdr.headers()?.clone();
    let header_len = header.len();
    let vocab_size = header_len - 2; // exclude first (email id) and last (label) columns

    // Load all records into memory so we can split them
    let all_records: Vec<StringRecord> = rdr.records().filter_map(|r| r.ok()).collect();
    let midpoint = all_records.len() / 2;
    let (train_set, test_set) = all_records.split_at(midpoint);

    // --- Training phase: learn from the first half ---
    let mut spam_totals = vec![0u32; header_len];
    let mut ham_totals = vec![0u32; header_len];
    let mut spam_count = 0u32;
    let mut ham_count = 0u32;

    for record in train_set {
        let is_spam = record.get(header_len - 1) == Some("1");
        if is_spam {
            spam_count += 1;
        } else {
            ham_count += 1;
        }
        let bucket = if is_spam {
            &mut spam_totals
        } else {
            &mut ham_totals
        };
        for i in 1..(header_len - 1) {
            let count: u32 = record[i].parse().unwrap_or(0);
            bucket[i] += count;
        }
    }

    let total_train = (spam_count + ham_count) as f64;
    let log_prior_spam = (spam_count as f64 / total_train).ln();
    let log_prior_ham = (ham_count as f64 / total_train).ln();

    let total_spam_words: u32 = spam_totals[1..(header_len - 1)].iter().sum();
    let total_ham_words: u32 = ham_totals[1..(header_len - 1)].iter().sum();

    // Laplace-smoothed log-likelihoods per word
    let log_likelihood_spam: Vec<f64> = spam_totals[1..(header_len - 1)]
        .iter()
        .map(|&x| ((x as f64 + 1.0) / (total_spam_words as f64 + vocab_size as f64)).ln())
        .collect();

    let log_likelihood_ham: Vec<f64> = ham_totals[1..(header_len - 1)]
        .iter()
        .map(|&x| ((x as f64 + 1.0) / (total_ham_words as f64 + vocab_size as f64)).ln())
        .collect();

    println!(
        "Training: {} emails ({} spam, {} ham)",
        train_set.len(),
        spam_count,
        ham_count
    );
    println!(
        "Log-priors: spam={:.4}, ham={:.4}",
        log_prior_spam, log_prior_ham
    );

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

        let (predicted, _words) = naive_bayes_classify(
            &header,
            log_prior_spam,
            log_prior_ham,
            &log_likelihood_spam,
            &log_likelihood_ham,
            &word_counts,
        );

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

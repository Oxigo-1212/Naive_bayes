use csv::{ReaderBuilder, StringRecord};
use std::error::Error;
use std::path::Path;
use std::vec;

fn naive_bayes_classify<'a>(
    headers: &'a StringRecord,
    log_prior_spam: f64,
    log_prior_ham: f64,
    log_likelihood_spam: &[f64],
    log_likelihood_ham: &[f64],
    spam_word_counts: &[u32],
    ham_word_counts: &[u32],
) -> (&'static str, Vec<&'a str>) {
    let spam_score = log_prior_spam
        + spam_word_counts
            .iter()
            .zip(log_likelihood_spam)
            .map(|(&count, &log_prob)| count as f64 * log_prob)
            .sum::<f64>();

    let ham_score = log_prior_ham
        + ham_word_counts
            .iter()
            .zip(log_likelihood_ham)
            .map(|(&count, &log_prob)| count as f64 * log_prob)
            .sum::<f64>();

    let label = if spam_score > ham_score {
        "spam"
    } else {
        "ham"
    };

    let words = spam_word_counts
        .iter()
        .zip(ham_word_counts.iter())
        .enumerate()
        .filter(|&(_, (&s_count, &h_count))| s_count > 0 || h_count > 0)
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
    let mut spam_totals = vec![0u32; header_len];
    let mut ham_totals = vec![0u32; header_len];
    let mut spam_counts = 0;
    let mut ham_counts = 0;
    for record in rdr.records() {
        let result = record?;
        let is_spam = result.get(header_len - 1) == Some("1");
        match is_spam {
            true => spam_counts += 1,
            false => ham_counts += 1,
        }
        let target_bucket = if is_spam {
            &mut spam_totals
        } else {
            &mut ham_totals
        };
        for i in 1..(header_len - 1) {
            let count: u32 = result[i].parse().unwrap_or(0);
            target_bucket[i] += count;
        }
    }
    let total_emails = (spam_counts + ham_counts) as f64;
    let prob_ham = ham_counts as f64 / total_emails;
    let prob_spam = spam_counts as f64 / total_emails;
    let total_spam_words: u32 = spam_totals.iter().sum();
    let total_ham_words: u32 = ham_totals.iter().sum();
    let vocab_size = header_len - 2;

    let _prob_conditional_spam: Vec<f64> = spam_totals
        .iter()
        .skip(1)
        .take(vocab_size)
        .map(|&x| {
            let p = (x as f64 + 1.0) / (total_spam_words as f64 + vocab_size as f64);
            p.ln()
        })
        .collect();
    let prob_conditional_ham: Vec<f64> = ham_totals
        .iter()
        .skip(1)
        .take(vocab_size)
        .map(|&x| {
            let p = (x as f64 + 1.0) / (total_ham_words as f64 + vocab_size as f64);
            p.ln()
        })
        .collect();
    let classification = naive_bayes_classify(
        &header,
        prob_spam.ln(),
        prob_ham.ln(),
        &vec![0.0; vocab_size], // Placeholder for spam likelihoods
        &prob_conditional_ham,
        &spam_totals[1..(header_len - 1)],
        &ham_totals[1..(header_len - 1)],
    );
    for word in classification.1.iter() {
        println!("{}: {}", word, classification.0);
    }
    Ok(())
}

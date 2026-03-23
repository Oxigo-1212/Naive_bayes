use csv::{ReaderBuilder, StringRecord};
use std::error::Error;
use std::path::Path;

pub struct NaiveBayesModel {
    pub headers: Vec<String>,
    pub log_prior_spam: f64,
    pub log_prior_ham: f64,
    pub log_likelihood_spam: Vec<f64>,
    pub log_likelihood_ham: Vec<f64>,
}

pub fn train<P: AsRef<Path>>(path: P) -> Result<NaiveBayesModel, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let header = rdr.headers()?.clone();
    let header_len = header.len();

    if header_len < 3 {
        return Err("CSV must include id, at least one word column, and label".into());
    }

    let vocab_size = header_len - 2; // exclude first (email id) and last (label) columns

    // Load all records into memory so we can split them
    let all_records: Vec<StringRecord> = rdr.records().collect::<Result<Vec<_>, csv::Error>>()?;

    if all_records.is_empty() {
        return Err("CSV contains no data rows".into());
    }

    let midpoint = all_records.len() / 2;
    let (train_set, _) = all_records.split_at(midpoint);

    if train_set.is_empty() {
        return Err("Training split is empty".into());
    }

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
    if total_train == 0.0 {
        return Err("No training samples found".into());
    }

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

    Ok(NaiveBayesModel {
        headers: header.iter().map(|s| s.to_string()).collect(),
        log_prior_spam,
        log_prior_ham,
        log_likelihood_spam,
        log_likelihood_ham,
    })
}

impl NaiveBayesModel {
    pub fn classify<'a>(&'a self, word_counts: &[u32]) -> (&'static str, Vec<&'a str>) {
        let spam_score = self.log_prior_spam
            + word_counts
                .iter()
                .zip(&self.log_likelihood_spam)
                .map(|(&count, &log_prob)| count as f64 * log_prob)
                .sum::<f64>();

        let ham_score = self.log_prior_ham
            + word_counts
                .iter()
                .zip(&self.log_likelihood_ham)
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
            .map(|(i, _)| self.headers.get(i + 1).map(|s| s.as_str()).unwrap_or(""))
            .collect();

        (label, words)
    }
}

#[cfg(test)]
mod tests {
    use super::NaiveBayesModel;

    #[test]
    fn classify_returns_spam_when_spam_score_is_higher() {
        let model = NaiveBayesModel {
            headers: vec![
                "id".to_string(),
                "offer".to_string(),
                "meeting".to_string(),
                "label".to_string(),
            ],
            log_prior_spam: -0.2,
            log_prior_ham: -1.2,
            log_likelihood_spam: vec![-0.1, -0.3],
            log_likelihood_ham: vec![-1.0, -0.2],
        };

        let (label, words) = model.classify(&[2, 1]);

        assert_eq!(label, "spam");
        assert_eq!(words, vec!["offer", "meeting"]);
    }

    #[test]
    fn classify_builds_words_vector_from_non_zero_counts() {
        let model = NaiveBayesModel {
            headers: vec![
                "id".to_string(),
                "hello".to_string(),
                "promo".to_string(),
                "project".to_string(),
                "label".to_string(),
            ],
            log_prior_spam: -0.69,
            log_prior_ham: -0.69,
            log_likelihood_spam: vec![-0.5, -1.2, -1.1],
            log_likelihood_ham: vec![-1.0, -0.4, -0.3],
        };

        let (_label, words) = model.classify(&[0, 3, 1]);

        assert_eq!(words, vec!["promo", "project"]);
    }
}

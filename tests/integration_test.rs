use spam_filter::train;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_csv_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    path.push(format!("spam_filter_integration_{nanos}.csv"));
    path
}

#[test]
fn train_builds_model_from_csv_and_classifies() {
    let csv_data = "id,buy,meeting,project,label
1,3,0,0,1
2,0,2,3,0
3,2,0,1,1
4,0,3,2,0
";

    let csv_path = temp_csv_path();
    fs::write(&csv_path, csv_data).expect("failed to write temporary CSV file");

    let model = train(&csv_path).expect("train should succeed for valid CSV");

    assert_eq!(model.headers.len(), 5);
    assert_eq!(model.log_likelihood_spam.len(), 3);
    assert_eq!(model.log_likelihood_ham.len(), 3);
    assert!(model.log_prior_spam.is_finite());
    assert!(model.log_prior_ham.is_finite());

    let (label, _words) = model.classify(&[1, 0, 0]);
    assert!(matches!(label, "spam" | "ham"));

    let _ = fs::remove_file(csv_path);
}

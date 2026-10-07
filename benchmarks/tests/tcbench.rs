//! Small runnable gate for the shared corpus, known values and thread wiring.
use std::process::Command;

#[test]
fn tcbench_verifies_requested_budgets() {
    for threads in ["1", "4", "8", "12"] {
        let out = Command::new(env!("CARGO_BIN_EXE_tcbench"))
            .args([
                "verify",
                "--threads",
                threads,
                "--size",
                "0.001",
                "--dtype",
                "f64,c64",
            ])
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(stdout.contains(&format!(
            "requested={threads} pool={threads} budget={threads}"
        )));
        assert!(stdout.contains("all comparisons within tolerance"));
    }
    let out = Command::new(env!("CARGO_BIN_EXE_tcbench"))
        .args(["verify", "--threads", "0"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}

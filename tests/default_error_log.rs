//! Proves the default error reporter routes through the `log` facade when the
//! `log` cargo feature is enabled. Run with: `cargo test --features log`.
//!
//! Isolated in its own test binary because `log::set_logger` is process-global
//! and one-shot. We install a capturing logger, do NOT override the device's
//! error function (so the built-in `default_error_reporter` runs), trigger a
//! real embree error, and assert an `Error`-level record reached the logger.
#![cfg(feature = "log")]

mod common;

use std::sync::Mutex;

use log::{Level, Metadata, Record};

static RECORDS: Mutex<Vec<(Level, String)>> = Mutex::new(Vec::new());

struct CapturingLogger;

impl log::Log for CapturingLogger {
    fn enabled(&self, _: &Metadata) -> bool { true }

    fn log(&self, record: &Record) {
        RECORDS
            .lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger;

#[test]
fn default_reporter_emits_through_log_facade() {
    log::set_logger(&LOGGER).expect("logger installed once for this test binary");
    log::set_max_level(log::LevelFilter::Error);

    // Default reporter is installed (report_errors defaults to true); leave it in
    // place so it is the thing that emits.
    let device = common::device();
    common::trigger_embree_error(&device);

    let records = RECORDS.lock().unwrap();
    assert!(
        records
            .iter()
            .any(|(level, msg)| *level == Level::Error && msg.contains("embree")),
        "default reporter should emit an Error-level `embree: …` log record; got {records:?}"
    );
}

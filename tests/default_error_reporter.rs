//! Proves the default error-reporting safety harness: it is on by default, and
//! a user-installed handler replaces it and receives embree's asynchronous
//! errors.
//!
//! The `log`-routed default reporter itself is proven in `default_error_log.rs`
//! (which needs `--features log` and a capturing logger). Here we prove the
//! wiring: the harness defaults on, and `set_error_function` overrides it.
mod common;

use std::sync::{Arc, Mutex};

use embree::{Config, Error};

#[test]
fn report_errors_is_on_by_default() {
    assert!(
        Config::default().report_errors,
        "the default error reporter must be installed by default"
    );
}

#[test]
fn custom_handler_replaces_default_and_receives_errors() {
    // The device is created with the default reporter installed (report_errors =
    // true). Replacing it must redirect embree's errors to our closure instead of
    // stderr/log, so the capture filling proves both that the override works and
    // that the error channel is live.
    let device = common::device();

    let captured: Arc<Mutex<Vec<Error>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    device.set_error_function(move |error, _msg| sink.lock().unwrap().push(error));

    common::trigger_embree_error(&device);

    let got = captured.lock().unwrap();
    assert!(
        !got.is_empty(),
        "the installed handler must replace the default and receive embree's error"
    );
    assert_eq!(
        got[0],
        Error::INVALID_ARGUMENT,
        "the out-of-range buffer slot should report INVALID_ARGUMENT"
    );
}

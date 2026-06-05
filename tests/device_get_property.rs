//! Regression proof for `Device::get_property` where it must treat a
//! legitimately zero property value as success, not as an error.
//!
//! `rtcGetDeviceProperty` does not use its return value to signal failure:
//! many properties are booleans where `0` means *false* / *not supported*. The
//! old code mapped every `0` to `Err(get_error())`, conflating "feature
//! disabled" with "call failed". The fix follows the embree contract: clear the
//! error, call, check the error.
mod common;

use embree3::DeviceProperty;

#[test]
fn zero_valued_property_is_ok_not_err() {
    let device = common::device();

    // `BACKFACE_CULLING_ENABLED` reflects the `backface_culling` config option,
    // which defaults to off (0) on a device created without it.
    let culling = device.get_property(DeviceProperty::BACKFACE_CULLING_ENABLED);
    assert_eq!(
        culling,
        Ok(0),
        "a property that is legitimately 0 must be Ok(0), not an error"
    );
}

#[test]
fn nonzero_property_still_works() {
    let device = common::device();

    // The version is a large nonzero integer (e.g. 31305 for 3.13.5).
    let version = device
        .get_property(DeviceProperty::VERSION)
        .expect("VERSION query must succeed");
    assert!(version > 0, "embree version should be a positive integer");
}

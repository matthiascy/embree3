//! Regression test for the triangle_geometry SIGSEGV:
//! `TiledImage::write_to_flat_buffer` used to overrun the flat framebuffer when
//! the image size was not a multiple of the tile size (the guarding
//! `debug_assert!` is compiled out in release). Edge tiles must be clamped to
//! the image bounds.
use support::TiledImage;

/// Blit into an OVERSIZED buffer and return (highest byte written, correct flat
/// length).
fn blit_extent(w: u32, h: u32, tw: u32, th: u32) -> (usize, usize) {
    let mut img = TiledImage::new(w, h, tw, th);
    img.pixels.iter_mut().for_each(|p| *p = 0xFFFF_FFFF); // sentinel so any OOB write shows
    let flat_len = (w * h * 4) as usize; // what the real window framebuffer is sized to
    let mut buf = vec![0u8; flat_len + 16 * 1024];
    img.write_to_flat_buffer(&mut buf);
    let last_written = buf
        .iter()
        .rposition(|&b| b != 0)
        .map(|p| p + 1)
        .unwrap_or(0);
    (last_written, flat_len)
}

#[test]
fn non_tile_aligned_size_stays_in_bounds() {
    // 10x10 image, 8x8 tiles: edge tiles overhang. Must NOT write past 10*10*4.
    let (last, flat) = blit_extent(10, 10, 8, 8);
    assert!(
        last <= flat,
        "blit overran: wrote up to byte {last}, flat buffer is {flat} ({} OOB)",
        last.saturating_sub(flat)
    );
}

#[test]
fn tile_aligned_size_stays_in_bounds() {
    let (last, flat) = blit_extent(16, 16, 8, 8);
    assert!(
        last <= flat,
        "blit overran: wrote up to byte {last}, flat buffer is {flat}"
    );
}

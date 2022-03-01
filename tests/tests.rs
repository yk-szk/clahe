use clahe::{clahe, pad_image};
use imageproc::assert_pixels_eq_within;
use std::path::PathBuf;

#[test]
fn test_pad_image() {
    let test_dir = env!("CARGO_MANIFEST_DIR");
    let input_filename: PathBuf = [test_dir, "tests/input/mandrill.png"].iter().collect();
    let input = image::open(&input_filename).unwrap().to_luma8();
    let output = pad_image(&input, 4, 4, 4, 4);
    let expected_filename: PathBuf = [test_dir, "tests/output/mandrill_BORDER_4_REFLECT_101.png"]
        .iter()
        .collect();
    let expected = image::open(&expected_filename).unwrap().to_luma8();
    assert_pixels_eq_within!(output, expected, 0);
}

#[test]
fn test_clahe() {
    let test_dir = env!("CARGO_MANIFEST_DIR");
    let input_filename: PathBuf = [test_dir, "tests/input/mandrill.png"].iter().collect();
    let input = image::open(&input_filename).unwrap().to_luma8();
    let output = clahe(&input, 8, 8, 8).unwrap();
    // output.save("clahe.png").unwrap();
    let expected_filename: PathBuf = [test_dir, "tests/output/mandrill_CLAHE_(8,(8,8)).png"]
        .iter()
        .collect();
    let expected = image::open(&expected_filename).unwrap().to_luma8();
    #[cfg(target_arch = "x86_64")]
    {
        // tolerance is 0 for x86_64 because of specialized `round` function that implements round half to even
        assert_pixels_eq_within!(output, expected, 0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert_pixels_eq_within!(output, expected, 1);
    }
}

#[test]
fn test_clahe16() {
    let test_dir = env!("CARGO_MANIFEST_DIR");
    let input_filename: PathBuf = [test_dir, "tests/input/mandrill_16bit.png"]
        .iter()
        .collect();
    let input = image::open(&input_filename).unwrap().to_luma16();
    let output = clahe(&input, 8, 8, 8).unwrap();
    // output.save("clahe16.png").unwrap();
    let expected_filename: PathBuf = [test_dir, "tests/output/mandrill_16bit_CLAHE_(8,(8,8)).png"]
        .iter()
        .collect();
    let expected = image::open(&expected_filename).unwrap().to_luma8();
    assert_pixels_eq_within!(output, expected, 1); // TODO: exactly equal implementation
}

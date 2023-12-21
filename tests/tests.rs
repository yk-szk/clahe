use anyhow::Result;
use clahe::{clahe, pad_image};
use imageproc::assert_pixels_eq_within;
use std::path::PathBuf;

const INPUT_FILENAME: &str = "input/mandrill.png";

fn test_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

#[test]
fn test_pad_image() -> Result<()> {
    let test_dir = test_directory();
    let input_filename = test_dir.join(INPUT_FILENAME);
    let input = image::open(input_filename)?.to_luma8();
    let output = pad_image(&input, 4, 4, 4, 4);
    let expected_filename = test_dir.join("output/mandrill_BORDER_4_REFLECT_101.png");
    let expected = image::open(expected_filename)?.to_luma8();
    assert_pixels_eq_within!(output, expected, 0);
    Ok(())
}

#[test]
fn test_clahe() -> Result<()> {
    let test_dir = test_directory();
    let input_filename = test_dir.join(INPUT_FILENAME);
    let input = image::open(input_filename)?.to_luma8();
    let output = clahe(&input, 8, 8, 8);
    let expected_filename = test_dir.join("output/mandrill_CLAHE_(8,(8,8)).png");
    let expected = image::open(expected_filename)?.to_luma8();
    #[cfg(target_arch = "x86_64")]
    {
        // tolerance is 0 for x86_64 because of specialized `round` function that implements round half to even
        assert_pixels_eq_within!(output, expected, 0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        assert_pixels_eq_within!(output, expected, 1);
    }
    Ok(())
}

#[test]
fn test_clahe16() -> Result<()> {
    let test_dir = test_directory();
    let input_filename = test_dir.join("input/mandrill_16bit.png");
    let input = image::open(input_filename)?.to_luma16();
    let output = clahe(&input, 8, 8, 8);
    let expected_filename = test_dir.join("output/mandrill_16bit_CLAHE_(8,(8,8)).png");
    let expected = image::open(expected_filename)?.to_luma16();
    assert_pixels_eq_within!(output, expected, 0);
    Ok(())
}

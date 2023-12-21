use clahe::clahe;
use std::path::PathBuf;

#[macro_use]
extern crate log;
use anyhow::{bail, Result};
use clap::Parser;
use env_logger::Builder;
use image::DynamicImage;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input: PathBuf,
    /// Output image filename
    output: PathBuf,

    /// Grid width
    #[clap(long = "width", default_value_t = 8)]
    grid_width: u32,
    /// Grid height
    #[clap(long = "height", default_value_t = 8)]
    grid_height: u32,
    /// Clip limit
    #[clap(long = "limit", default_value_t = 40)]
    clip_limit: u32,

    /// Force 8 bits output even for 16 bits input
    #[clap(short, long)]
    eight: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    Builder::default()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    info!("Load image");
    let im = image::open(args.input)?;

    info!("CLAHE");
    let output = match im {
        DynamicImage::ImageLuma8(img) => {
            debug!("u8 input");
            DynamicImage::ImageLuma8(clahe(
                &img,
                args.grid_width,
                args.grid_height,
                args.clip_limit,
            ))
        }
        DynamicImage::ImageLuma16(img) => {
            if args.eight {
                debug!("u16 input and u8 output");
                DynamicImage::ImageLuma8(clahe(
                    &img,
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                ))
            } else {
                debug!("u16 input and u16 output");
                DynamicImage::ImageLuma16(clahe(
                    &img,
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                ))
            }
        }
        DynamicImage::ImageRgb8(img) => {
            debug!("rgb input");
            info!("Convert rgb to gray");
            let luma = image::GrayImage::from_fn(img.width(), img.height(), |x, y| {
                image::Luma([img.get_pixel(x, y).0[0]])
            });
            DynamicImage::ImageLuma8(clahe(
                &luma,
                args.grid_width,
                args.grid_height,
                args.clip_limit,
            ))
        }
        _ => {
            bail!("u8, u16, rgb8, or rgba16 image is expected");
        }
    };

    info!("Save {:?}", args.output);
    output.save(&args.output)?;
    Ok(())
}

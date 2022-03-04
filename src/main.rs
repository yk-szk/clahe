use clahe::clahe;
use std::path::Path;

#[macro_use]
extern crate log;
use clap::Parser;
use env_logger::{Builder, Env};
use image::DynamicImage;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Input image filename
    input: String,
    /// Output image filename
    output: String,

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
    /// Set verbosity
    #[clap(short, long, parse(from_occurrences))]
    verbose: usize,
}

fn main() {
    let args = Args::parse();
    let log_level = if args.verbose == 0 {
        "error"
    } else if args.verbose == 1 {
        "info"
    } else {
        "debug"
    };
    let env = Env::default().filter_or("LOG_LEVEL", log_level);
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    info!("Load image");
    let im = image::open(&Path::new(&args.input)).unwrap();

    info!("CLAHE");
    let output = match im {
        DynamicImage::ImageLuma8(img) => {
            debug!("u8 input");
            DynamicImage::ImageLuma8(
                clahe(&img, args.grid_width, args.grid_height, args.clip_limit).unwrap(),
            )
        }
        DynamicImage::ImageLuma16(img) => {
            if args.eight {
                debug!("u16 input and u8 output");
                DynamicImage::ImageLuma8(
                    clahe(&img, args.grid_width, args.grid_height, args.clip_limit).unwrap(),
                )
            } else {
                debug!("u16 input and u16 output");
                DynamicImage::ImageLuma16(
                    clahe(&img, args.grid_width, args.grid_height, args.clip_limit).unwrap(),
                )
            }
        }
        DynamicImage::ImageRgb8(img) => {
            debug!("rgb input");
            info!("Convert rgb to gray");
            let luma = image::GrayImage::from_fn(img.width(), img.height(), |x, y| {
                image::Luma([img.get_pixel(x, y).0[0]])
            });
            DynamicImage::ImageLuma8(
                clahe(&luma, args.grid_width, args.grid_height, args.clip_limit).unwrap(),
            )
        }
        _ => {
            panic!("u8, u16, rgb8, or rgba16 image is expected");
        }
    };

    info!("Save {}", args.output);
    output.save(&Path::new(&args.output)).unwrap();
}

use clahe::clahe;
use std::path::Path;

#[macro_use]
extern crate log;
use clap::Parser;
use env_logger::{Builder, Env};
use image::error::{ImageFormatHint, UnsupportedError};
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
            clahe(&img, args.grid_width, args.grid_height, args.clip_limit)
        }
        DynamicImage::ImageLuma16(img) => {
            debug!("u16 input");
            clahe(&img, args.grid_width, args.grid_height, args.clip_limit)
        }
        _ => {
            let hint =
                ImageFormatHint::Name("u8, u16, rgb8, or rgba16 image is expected".to_string());
            Err(UnsupportedError::from(hint).into())
        }
    }
    .unwrap();

    info!("Save {}", args.output);
    output.save(&Path::new(&args.output)).unwrap();
}

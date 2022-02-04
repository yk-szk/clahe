use std::path::Path;
use clahe::clahe;

#[macro_use]
extern crate log;
use clap::Parser;
use env_logger::{Builder, Env};

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
}

fn main() {
    let args = Args::parse();

    let env = Env::default().filter_or("LOG_LEVEL", "debug");
    Builder::from_env(env)
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    info!("Load image");
    let im = image::open(&Path::new(&args.input)).unwrap();

    info!("CLAHE");
    let output = clahe(
        &im.to_luma8(),
        args.grid_width,
        args.grid_height,
        args.clip_limit,
    )
    .unwrap();

    info!("save {}", args.output);
    output.save(&Path::new(&args.output)).unwrap();
}

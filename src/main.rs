use clahe::clahe_image;
use std::path::PathBuf;

#[macro_use]
extern crate log;
use anyhow::{bail, Context, Result};
use clap::Parser;
use image::DynamicImage;
use ndarray::prelude::*;

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
    /// Sampling rate for tile
    #[clap(long = "sample", default_value_t = 1.0)]
    tile_sample: f64,

    /// Force 8 bits output even for 16 bits input
    #[clap(short, long)]
    eight: bool,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    let im = image::open(&args.input).with_context(|| format!("Loading {:?}", args.input))?;

    info!("CLAHE");
    let output = match im {
        DynamicImage::ImageLuma8(img) => {
            debug!("u8 input");
            DynamicImage::ImageLuma8(clahe_image(
                &img,
                args.grid_width,
                args.grid_height,
                args.clip_limit,
                args.tile_sample,
            )?)
        }
        DynamicImage::ImageLuma16(img) => {
            if args.eight {
                debug!("u16 input and u8 output");
                DynamicImage::ImageLuma8(clahe_image(
                    &img,
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                    args.tile_sample,
                )?)
            } else {
                debug!("u16 input and u16 output");
                DynamicImage::ImageLuma16(clahe_image(
                    &img,
                    args.grid_width,
                    args.grid_height,
                    args.clip_limit,
                    args.tile_sample,
                )?)
            }
        }
        DynamicImage::ImageRgb8(img) => {
            debug!("rgb input");
            let image_shape = (img.height() as usize, img.width() as usize);
            let arr_rgb = Array2::from_shape_vec(
                (img.height() as usize * img.width() as usize, 3usize),
                img.into_vec(),
            )
            .context("Image to array")?;
            let mut arr_hsl = arr_rgb.map_axis(Axis(1), |rgb| {
                let rgb = coolor::Rgb::new(rgb[0], rgb[1], rgb[2]);
                rgb.to_hsl()
            });
            let arr_lum = arr_hsl.map(|hsl| (hsl.l * 255.0).round() as u8);
            let arr_lum = Array2::from_shape_vec(image_shape, arr_lum.into_raw_vec())
                .context("Rgb array to luminosity array")?;
            let new_lum: Array2<u8> = clahe::clahe_ndarray(
                arr_lum.view(),
                args.grid_width,
                args.grid_height,
                args.clip_limit,
                args.tile_sample,
            )?;
            let new_lum = Array1::from_shape_vec(new_lum.len(), new_lum.into_raw_vec())?;
            arr_hsl.zip_mut_with(&new_lum, |hsl, lum| {
                hsl.l = *lum as f32 / 255.0;
            });

            // Use `Vec` as intermediate container because I could not figure out
            // a way to directly convert Array to Image
            let mut vec_rgb = arr_rgb.into_raw_vec(); //vec![0; image_shape.0 * image_shape.1 * 3];
            vec_rgb.clear();
            arr_hsl.for_each(|hsl| {
                let rgb = hsl.to_rgb();
                vec_rgb.push(rgb.r);
                vec_rgb.push(rgb.g);
                vec_rgb.push(rgb.b);
            });

            let rgb_img =
                image::RgbImage::from_raw(image_shape.1 as u32, image_shape.0 as u32, vec_rgb)
                    .unwrap();
            DynamicImage::ImageRgb8(rgb_img)
        }
        invalid_image => {
            bail!(
                "L8, L16, or RGB image is expected. Found {:?}",
                invalid_image.color()
            );
        }
    };

    output
        .save(&args.output)
        .with_context(|| format!("Saving {:?}", args.output))?;
    Ok(())
}

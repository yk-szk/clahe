use std::usize;

use image::*;
#[macro_use]
extern crate log;

type HistType = [u32];

#[allow(dead_code)]
fn calc_tile_hist<T>(
    img: &ImageBuffer<Luma<T>, Vec<T>>,
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    hist: &mut HistType,
) where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + 'static,
{
    hist.fill(0);
    let (full_width, _full_height) = img.dimensions();

    let img_raw = img.as_raw();
    unsafe {
        for y in top..(top + height) {
            let offset = (y * full_width) as usize;
            for index in (offset + left as usize)..(offset + left as usize + width as usize) {
                let pix = *img_raw.get_unchecked(index);
                let hist_index: usize = pix.into();
                *hist.get_unchecked_mut(hist_index) += 1;
            }
        }
    }
}

// fn calc_tile_hist_ndarray<S>(
//     img: ArrayBase<S, Ix2>,
//     left: u32,
//     top: u32,
//     width: u32,
//     height: u32,
//     hist: &mut HistType,
// ) where
//     S: ndarray::Data + ndarray::RawData,
//     S::Elem: std::clone::Clone + Default,
// {
//     hist.fill(0);
//     let (full_width, _full_height) = img.dim();

//     let img_raw = img.into_raw_vec();
//     unsafe {
//         for y in top..(top + height) {
//             let offset = (y * full_width) as usize;
//             for index in (offset + left as usize)..(offset + left as usize + width as usize) {
//                 let pix = *img_raw.get_unchecked(index);
//                 let hist_index: usize = pix.into();
//                 *hist.get_unchecked_mut(hist_index) += 1;
//             }
//         }
//     }
// }

fn clip_hist(hist: &mut HistType, limit: u32) {
    let mut clipped: u32 = 0;

    hist.iter_mut().for_each(|count| {
        if *count > limit {
            clipped += *count - limit;
            *count = limit;
        }
    });

    if clipped > 0 {
        let hist_size = hist.len() as u32;
        let redist_batch = clipped / hist_size;

        if redist_batch > 0 {
            hist.iter_mut().for_each(|count| {
                *count += redist_batch;
            });
        }

        let residual = (clipped - redist_batch * hist_size) as usize;
        if residual > 0 {
            let step = usize::max(1, hist_size as usize / residual);
            let end = step * residual;
            hist.iter_mut().take(end).step_by(step).for_each(|count| {
                *count += 1;
            });
        }
    }
}

#[allow(dead_code)]
fn clip_hist_g<T>(hist: &mut [T], limit: T)
where
    T: PartialOrd
        + Copy
        + num_traits::FromPrimitive
        + num_traits::NumAssign
        + num_traits::NumOps
        + num_traits::Zero
        + num_traits::One,
    usize: From<T>,
{
    let mut clipped: usize = 0;

    hist.iter_mut().for_each(|count| {
        if *count > limit {
            clipped += usize::from(*count - limit);
            *count = limit;
        }
    });

    if clipped > 0 {
        let hist_size = hist.len();
        let redist_batch = clipped / hist_size;

        if redist_batch > 0 {
            hist.iter_mut().for_each(|count| {
                *count += T::from_usize(redist_batch).unwrap();
            });
        }

        let residual = clipped - redist_batch * hist_size;
        if residual > 0 {
            let step = usize::max(1, hist_size / residual);
            let end = step * residual;
            hist.iter_mut().take(end).step_by(step).for_each(|count| {
                *count += T::one();
            });
        }
    }
}

/// Round half to even
#[cfg(target_arch = "x86_64")]
unsafe fn round(f: f32) -> i32 {
    let tmp = core::arch::x86_64::_mm_set_ss(f);
    core::arch::x86_64::_mm_cvtss_si32(tmp)
}
#[cfg(target_arch = "x86_64")]
type RoundOutput = i32;

/// f32::round
#[cfg(not(target_arch = "x86_64"))]
unsafe fn round(f: f32) -> f32 {
    f.round()
}
#[cfg(not(target_arch = "x86_64"))]
type RoundOutput = f32;

pub trait RoundFrom {
    fn round_from(f: f32) -> Self;
}

impl RoundFrom for u8 {
    fn round_from(f: f32) -> Self {
        unsafe { round(f) as u8 }
    }
}

impl RoundFrom for u16 {
    fn round_from(f: f32) -> Self {
        unsafe { round(f) as u16 }
    }
}

/// Trait to cast i32/f32 to u8/u16.
/// Needed to make `calc_lut` generic. There maybe more idiomatic solution that doesn't require this trait.
pub trait CastFrom<T> {
    fn cast_from(o: RoundOutput) -> T;
}
impl CastFrom<u8> for u8 {
    fn cast_from(o: RoundOutput) -> u8 {
        o as u8
    }
}
impl CastFrom<u16> for u16 {
    fn cast_from(o: RoundOutput) -> u16 {
        o as u16
    }
}

fn calc_lut<T: RoundFrom>(hist: &HistType, lut: &mut [T], scale: f32) {
    let mut cumsum: u32 = 0;
    unsafe {
        for index in 0..hist.len() {
            cumsum += *hist.get_unchecked(index);
            *lut.get_unchecked_mut(index) = T::round_from(cumsum as f32 * scale);
        }
    }
}

use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

// fn _clahe_ndarray<S>(
//     (original_input_width, original_input_height): (u32, u32),
//     input: ArrayBase<S, Ix2>,
//     grid_width: u32,
//     grid_height: u32,
//     clip_limit: u32,
//     tile_sample: f64,
// ) -> Array2<S>
// where
//     S: ndarray::DataMut
//         + ndarray::Data
//         + ndarray::DataMut
//         + ndarray::Data
//         + ndarray::RawData
//         + num_traits::Zero,
//     // S::Elem: std::clone::Clone + Default,
// {
//     let output = Array2::<S>::zeros((
//         original_input_width as usize,
//         original_input_height as usize,
//     ));
//     output
// }
// use num_traits::bounds::UpperBounded;

fn _clahe_ndarray<A, B, S, T>(
    (original_input_width, original_input_height): (u32, u32),
    input: ArrayBase<S, Ix2>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> ArrayBase<T, Ix2>
where
    S: Clone + ndarray::Data<Elem = A>,
    A: Copy + PartialOrd,
    T: Clone + ndarray::DataOwned<Elem = B> + ndarray::RawDataMut<Elem = B>,
    B: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    f32: From<A> + From<B>,
    u32: From<A>,
    usize: From<A>,
{
    let (input_width, input_height) = (input.ncols() as u32, input.nrows() as u32);
    debug!("Input size {input_width} x {input_height}");
    debug!("Grid size {} x {}", grid_width, grid_height);
    let tile_width = original_input_width / grid_width;
    let tile_height = original_input_height / grid_height;
    debug!("Tile size {} x {}", tile_width, tile_height);

    let mut output = ArrayBase::zeros((
        original_input_height as usize,
        original_input_width as usize,
    ));

    let max_pix_value = input.max().unwrap();

    // max_pixe_value + 1 is used as the size of the histogram to reduce the computation for clip_hist and calc_lut.
    // This is different from OpenCV's size (T::Max + 1).
    // This difference does not affect test images in tests directories,
    let hist_size: usize = usize::max(u8::MAX as usize, usize::from(*max_pix_value)) + 1;
    debug!("Hist size {}", hist_size);
    let lut_size = hist_size as u32;
    let lut_scale = f32::from(B::max_value()) / (tile_width * tile_height) as f32;

    let clip_limit = if clip_limit > 0 {
        let new_limit = u32::max(
            1,
            clip_limit * (tile_width * tile_height) / hist_size as u32,
        ); // OpenCV does this.
        debug!("New clip limit {}", new_limit);
        new_limit
    } else {
        0
    };
    let tile_step_width = tile_width / tile_sample as u32;
    let tile_step_height = tile_height / tile_sample as u32;
    let sampled_grid_width = (input_width - tile_width) / tile_step_width + 1;
    let sampled_grid_height = (input_height - tile_height) / tile_step_height + 1;
    // let sampled_grid_width = input_width / tile_step_width;
    // let sampled_grid_height = input_height / tile_step_height;
    // let sampled_grid_width = (grid_width as f64 * tile_sample).ceil() as u32;
    // let sampled_grid_height = (grid_height as f64 * tile_sample).ceil() as u32;
    debug!(
        "Sampled grid size {} x {}",
        sampled_grid_width, sampled_grid_height
    );

    debug!("Tile step size {} x {}", tile_step_width, tile_step_height);

    debug!("Calculate lookup tables");
    let mut lookup_tables: Vec<B> =
        vec![B::zero(); (sampled_grid_height * sampled_grid_width * lut_size) as usize];
    let mut hist = vec![0; hist_size];
    unsafe {
        for slice_idx in 0..sampled_grid_height {
            let slice = &mut lookup_tables[(slice_idx * sampled_grid_width * lut_size) as usize
                ..((slice_idx + 1) * sampled_grid_width * lut_size) as usize];
            // println!("top: {}", tile_step_height * slice_idx);
            for row_idx in 0..sampled_grid_width {
                let lut =
                    &mut slice[(row_idx * lut_size) as usize..((row_idx + 1) * lut_size) as usize];

                let (left, top, width, height) = (
                    (tile_step_width * row_idx) as usize,
                    (tile_step_height * slice_idx) as usize,
                    tile_width as usize,
                    tile_height as usize,
                );
                hist.fill(0);

                input
                    .slice(s![top..top + height, left..left + width])
                    .iter()
                    .for_each(|pix| {
                        let hist_index: usize = (*pix).into();
                        hist[hist_index] += 1;
                    });

                // calc_tile_hist(input, left, top, width, height, hist.as_mut_slice());
                if clip_limit >= 1 {
                    clip_hist(hist.as_mut_slice(), clip_limit);
                }
                calc_lut(hist.as_mut_slice(), lut, lut_scale);
            }
        }
        type Float = f32;

        debug!("Apply interpolations");
        let output_ptr: *mut B = output.as_mut_ptr();

        // pre calculate x positions and weights
        let lr_luts_x_weights = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        info!(
            "Max lut index {}",
            lr_luts_x_weights.last().unwrap().1 / lut_size
        );
        // perform interpolation
        for y in 0..(original_input_height as usize) {
            let (top_y, bottom_y, y_weight) =
                calculate_lut_weights_for_position(y, tile_step_height, sampled_grid_height, 1);
            let output_row_ptr = output_ptr.add(y * original_input_width as usize);
            let top_lut = &lookup_tables[(top_y * sampled_grid_width * lut_size) as usize
                ..((top_y + 1) * sampled_grid_width * lut_size) as usize];
            let bottom_lut = &lookup_tables[(bottom_y * sampled_grid_width * lut_size) as usize
                ..((bottom_y + 1) * sampled_grid_width * lut_size) as usize];
            for x in 0..(original_input_width as usize) {
                let input_pixel: u32 = (*input.get((y, x)).unwrap()).into();
                let (left, right, x_weight) = *lr_luts_x_weights.get_unchecked(x);

                let top_left = *top_lut.get_unchecked((input_pixel + left) as usize);
                let bottom_left = *bottom_lut.get_unchecked((input_pixel + left) as usize);
                let top_right = *top_lut.get_unchecked((input_pixel + right) as usize);
                let bottom_right = *bottom_lut.get_unchecked((input_pixel + right) as usize);

                #[inline]
                fn interpolate<T: Into<Float>>(left: T, right: T, right_weight: Float) -> Float {
                    let left: Float = left.into();
                    let right: Float = right.into();
                    left as Float * (1.0 - right_weight) + right as Float * right_weight
                }
                let intermediate_1 = interpolate(top_left, top_right, x_weight);
                let intermediate_2 = interpolate(bottom_left, bottom_right, x_weight);
                let interpolated = interpolate(intermediate_1, intermediate_2, y_weight);
                let interpolated = B::round_from(interpolated);
                *output_row_ptr.add(x) = interpolated;
            }
        }
    }

    output
}

// fn _type_test<A, B, S, T>(
//     (original_input_width, original_input_height): (u32, u32),
//     input: ArrayBase<S, Ix2>,
// ) -> ArrayBase<T, Ix2>
// where
//     S: Clone + ndarray::Data<Elem = A>,
//     A: Copy + PartialOrd,
//     T: Clone + Default + ndarray::Data<Elem = B> + num_traits::Zero + RoundFrom,
//     B: num_traits::Bounded + RoundFrom,
//     f32: From<S::Elem> + From<T::Elem> + From<T>,
//     u32: From<S::Elem>,
//     usize: From<S::Elem>,
// {
//     let mut output = Array2::<T>::zeros((
//         original_input_width as usize,
//         original_input_height as usize,
//     ));
//     output
// }
// use ndarray::prelude::*;

pub fn clahe_ndarray<S, T>(
    input: ArrayView2<S>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> Array2<T>
where
    S: Copy + PartialOrd + num_traits::Zero,
    T: num_traits::Bounded + RoundFrom + Clone + Copy + num_traits::Zero,
    f32: From<S> + From<T>,
    u32: From<S>,
    usize: From<S>,
{
    let (input_width, input_height) = (input.ncols() as u32, input.nrows() as u32);
    let tile_width = input_width / grid_width;
    let tile_height = input_height / grid_height;

    // let sampled_grid_width = (grid_width as f64 * tile_sample).ceil() as u32;
    // let sampled_grid_height = (grid_height as f64 * tile_sample).ceil() as u32;
    let tile_step_width = tile_width / tile_sample as u32;
    let tile_step_height = tile_height / tile_sample as u32;
    let sampled_grid_width = (input_width - tile_width) / tile_step_width + 1;
    let sampled_grid_height = (input_height - tile_height) / tile_step_height + 1;
    // let sampled_grid_width = input_width / tile_step_width;
    // let sampled_grid_height = input_height / tile_step_height;

    if input_width % sampled_grid_width != 0 || input_height % grid_height != 0 {
        let pad_width =
            (sampled_grid_width - input_width % sampled_grid_width) % sampled_grid_width;
        let pad_height =
            (sampled_grid_height - input_height % sampled_grid_height) % sampled_grid_height;
        debug!(
            "Padding image by {} in width and {} in height",
            pad_width, pad_height
        );
        let padded: Array2<S> = pad_array(input, 0, pad_height as usize, 0, pad_width as usize);
        _clahe_ndarray(
            (input_width, input_height),
            padded,
            grid_width,
            grid_height,
            clip_limit,
            tile_sample,
        )
    } else {
        _clahe_ndarray(
            (input_width, input_height),
            input,
            grid_width,
            grid_height,
            clip_limit,
            tile_sample,
        )
    }
}

/// CLAHE implementation.
/// Input shape must be divisible by grid_width and grid_height.
///
/// # Arguments
/// * `original_input_width` - original input width
/// * `original_input_height` - original input height
/// * `input` - GrayImage or Gray16Image (padded if needed)
fn _clahe<T, S>(
    (original_input_width, original_input_height): (u32, u32),
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> ImageBuffer<Luma<S>, Vec<S>>
where
    T: image::Primitive
        + Into<usize>
        + Into<u32>
        + Ord
        + RoundFrom
        + CastFrom<T>
        + Default
        + 'static,
    S: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + CastFrom<S> + 'static,
    f32: From<T> + From<S>,
{
    debug!("Original size {original_input_width} x {original_input_height}");

    let (input_width, input_height) = input.dimensions();
    debug!("Input size {input_width} x {input_height}");
    debug!("Grid size {} x {}", grid_width, grid_height);
    let tile_width = original_input_width / grid_width;
    let tile_height = original_input_height / grid_height;
    debug!("Tile size {} x {}", tile_width, tile_height);
    let max_pix_value = *input.iter().max().unwrap();

    // max_pixe_value + 1 is used as the size of the histogram to reduce the computation for clip_hist and calc_lut.
    // This is different from OpenCV's size (T::Max + 1).
    // This difference does not affect test images in tests directories,
    let hist_size: usize = usize::max(u8::MAX as usize, max_pix_value.into()) + 1;
    debug!("Hist size {}", hist_size);
    let lut_size = hist_size as u32;
    let lut_scale = f32::from(S::max_value()) / (tile_width * tile_height) as f32;

    let clip_limit = if clip_limit > 0 {
        let new_limit = u32::max(
            1,
            clip_limit * (tile_width * tile_height) / hist_size as u32,
        ); // OpenCV does this.
        debug!("New clip limit {}", new_limit);
        new_limit
    } else {
        0
    };
    let tile_step_width = tile_width / tile_sample as u32;
    let tile_step_height = tile_height / tile_sample as u32;
    let sampled_grid_width = (input_width - tile_width) / tile_step_width + 1;
    let sampled_grid_height = (input_height - tile_height) / tile_step_height + 1;
    // let sampled_grid_width = input_width / tile_step_width;
    // let sampled_grid_height = input_height / tile_step_height;
    // let sampled_grid_width = (grid_width as f64 * tile_sample).ceil() as u32;
    // let sampled_grid_height = (grid_height as f64 * tile_sample).ceil() as u32;
    debug!(
        "Sampled grid size {} x {}",
        sampled_grid_width, sampled_grid_height
    );

    debug!("Tile step size {} x {}", tile_step_width, tile_step_height);

    let mut output =
        ImageBuffer::<Luma<S>, Vec<S>>::new(original_input_width, original_input_height);

    debug!("Calculate lookup tables");
    let mut lookup_tables: Vec<T> =
        vec![T::default(); (sampled_grid_height * sampled_grid_width * lut_size) as usize];
    let mut hist = vec![0; hist_size];
    unsafe {
        for slice_idx in 0..sampled_grid_height {
            let slice = &mut lookup_tables[(slice_idx * sampled_grid_width * lut_size) as usize
                ..((slice_idx + 1) * sampled_grid_width * lut_size) as usize];
            // println!("top: {}", tile_step_height * slice_idx);
            for row_idx in 0..sampled_grid_width {
                let lut =
                    &mut slice[(row_idx * lut_size) as usize..((row_idx + 1) * lut_size) as usize];

                let (left, top, width, height) = (
                    tile_step_width * row_idx,
                    tile_step_height * slice_idx,
                    tile_width,
                    tile_height,
                );

                calc_tile_hist(input, left, top, width, height, hist.as_mut_slice());
                if clip_limit >= 1 {
                    clip_hist(hist.as_mut_slice(), clip_limit);
                }
                calc_lut(hist.as_mut_slice(), lut, lut_scale);
            }
        }
        type Float = f32;

        debug!("Apply interpolations");
        let output_ptr = output.as_mut_ptr();

        // pre calculate x positions and weights
        let lr_luts_x_weights = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        info!(
            "Max lut index {}",
            lr_luts_x_weights.last().unwrap().1 / lut_size
        );
        // perform interpolation
        for y in 0..(original_input_height as usize) {
            let (top_y, bottom_y, y_weight) =
                calculate_lut_weights_for_position(y, tile_step_height, sampled_grid_height, 1);
            let output_row_ptr = output_ptr.add(y * original_input_width as usize);
            let top_lut = &lookup_tables[(top_y * sampled_grid_width * lut_size) as usize
                ..((top_y + 1) * sampled_grid_width * lut_size) as usize];
            let bottom_lut = &lookup_tables[(bottom_y * sampled_grid_width * lut_size) as usize
                ..((bottom_y + 1) * sampled_grid_width * lut_size) as usize];
            for x in 0..(original_input_width as usize) {
                let input_pixel: u32 = input.unsafe_get_pixel(x as u32, y as u32).0[0].into();
                let (left, right, x_weight) = *lr_luts_x_weights.get_unchecked(x);

                let top_left = *top_lut.get_unchecked((input_pixel + left) as usize);
                let bottom_left = *bottom_lut.get_unchecked((input_pixel + left) as usize);
                let top_right = *top_lut.get_unchecked((input_pixel + right) as usize);
                let bottom_right = *bottom_lut.get_unchecked((input_pixel + right) as usize);

                #[inline]
                fn interpolate<T: Into<Float>>(left: T, right: T, right_weight: Float) -> Float {
                    let left: Float = left.into();
                    let right: Float = right.into();
                    left as Float * (1.0 - right_weight) + right as Float * right_weight
                }
                let intermediate_1 = interpolate(top_left, top_right, x_weight);
                let intermediate_2 = interpolate(bottom_left, bottom_right, x_weight);
                let interpolated = interpolate(intermediate_1, intermediate_2, y_weight);
                let interpolated = S::cast_from(round(interpolated));
                *output_row_ptr.add(x) = interpolated;
            }
        }
    }

    output
}

fn calculate_lut_and_weights(
    original_input_width: u32,
    _tile_width: u32,
    tile_step_width: u32,
    sampled_grid_width: u32,
    lut_size: u32,
) -> Vec<(u32, u32, f32)> {
    (0..(original_input_width as usize))
        .map(|x| {
            calculate_lut_weights_for_position(x, tile_step_width, sampled_grid_width, lut_size)
        })
        .collect::<Vec<_>>()
}

fn calculate_lut_weights_for_position(
    index: usize,
    step_size: u32,
    sampled_grid_size: u32,
    lut_dimension: u32,
) -> (u32, u32, f32) {
    let lut_position = index as f64 / step_size as f64 - 0.5;
    let lower_bound = lut_position.floor().min((sampled_grid_size - 1) as f64);
    let upper_bound = std::cmp::min((lower_bound + 1.0) as u32, sampled_grid_size - 1);
    let position_weight = 0.0f64.max(lut_position - lower_bound);
    let lower_bound = lower_bound as u32;
    (
        lower_bound * lut_dimension,
        upper_bound * lut_dimension,
        position_weight as f32,
    )
}

pub fn image2array_view2<T>(input: &ImageBuffer<Luma<T>, Vec<T>>) -> ArrayView2<T>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + 'static,
{
    let (input_width, input_height) = input.dimensions();
    unsafe {
        ArrayView2::from_shape_ptr(
            (input_width as usize, input_height as usize),
            input.as_ptr(),
        )
    }
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE)
/// Interpolation is performed for efficiency.
///
/// # Arguments
/// * `input` - GrayImage or Gray16Image
///
/// The CLAHE implementation is based on OpenCV, which is licensed under Apache 2 License.
pub fn clahe<T, S>(
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
    tile_sample: f64,
) -> ImageBuffer<Luma<S>, Vec<S>>
where
    T: image::Primitive
        + Into<usize>
        + Into<u32>
        + Ord
        + RoundFrom
        + CastFrom<T>
        + Default
        + 'static,
    S: image::Primitive + Into<usize> + Into<u32> + Ord + RoundFrom + CastFrom<S> + 'static,
    f32: From<T> + From<S>,
    u32: From<T>,
    usize: From<T>,
{
    let (input_width, input_height) = input.dimensions();
    let arr = clahe_ndarray(
        image2array_view2(input),
        grid_width,
        grid_height,
        clip_limit,
        tile_sample,
    );
    ImageBuffer::<Luma<S>, Vec<S>>::from_vec(input_width, input_height, arr.into_raw_vec()).unwrap()

    // let (input_width, input_height) = input.dimensions();
    // let tile_width = input_width / grid_width;
    // let tile_height = input_height / grid_height;

    // // let sampled_grid_width = (grid_width as f64 * tile_sample).ceil() as u32;
    // // let sampled_grid_height = (grid_height as f64 * tile_sample).ceil() as u32;
    // let tile_step_width = tile_width / tile_sample as u32;
    // let tile_step_height = tile_height / tile_sample as u32;
    // let sampled_grid_width = (input_width - tile_width) / tile_step_width + 1;
    // let sampled_grid_height = (input_height - tile_height) / tile_step_height + 1;
    // // let sampled_grid_width = input_width / tile_step_width;
    // // let sampled_grid_height = input_height / tile_step_height;

    // if input_width % sampled_grid_width != 0 || input_height % grid_height != 0 {
    //     let pad_width =
    //         (sampled_grid_width - input_width % sampled_grid_width) % sampled_grid_width;
    //     let pad_height =
    //         (sampled_grid_height - input_height % sampled_grid_height) % sampled_grid_height;
    //     debug!(
    //         "Padding image by {} in width and {} in height",
    //         pad_width, pad_height
    //     );
    //     let padded = pad_image(input, 0, pad_height, 0, pad_width);
    //     _clahe(
    //         input.dimensions(),
    //         &padded,
    //         grid_width,
    //         grid_height,
    //         clip_limit,
    //         tile_sample,
    //     )
    // } else {
    // let mut output = ImageBuffer::<Luma<S>, Vec<S>>::new(input_width, input_height);
    // output.copy_from(
    //     &ImageBuffer::<Luma<S>, Vec<S>>::from_raw(
    //         input_width,
    //         input_height,
    //         arr.into_raw_vec(),
    //     )
    //     .unwrap(),
    // );
    // }
}

/// Pad image (copyMakeBorder)
///
/// Compatible with OpenCV's copyMakeBorder with `borderType = cv2.BORDER_REFLECT_101`.
///
/// # Arguments
/// * `input` - input image
/// * `top`, `bottom`, `left`, `right` - padding sizes
///
/// # Examples
///
/// ```
/// use clahe::pad_image;
/// let input = imageproc::gray_image!(type: u8, 0,1,2; 3,4,5);
/// let output = pad_image(&input, 1, 1, 1, 1);
/// let expected = imageproc::gray_image!(type:u8, 4,3,4,5,4; 1,0,1,2,1; 4,3,4,5,4; 1,0,1,2,1);
/// assert_eq!(output, expected);
/// ```
pub fn pad_image<T>(
    input: &ImageBuffer<T, Vec<<T as Pixel>::Subpixel>>,
    top: u32,
    bottom: u32,
    left: u32,
    right: u32,
) -> ImageBuffer<T, Vec<<T as Pixel>::Subpixel>>
where
    T: image::Pixel + 'static,
{
    let (input_width, input_height) = input.dimensions();
    let (output_width, output_height) = (input_width + left + right, input_height + top + bottom);
    let mut output = ImageBuffer::new(output_width, output_height);
    for y in 0..input_height {
        for x in 0..input_width {
            unsafe {
                output.unsafe_put_pixel(x + left, y + top, input.unsafe_get_pixel(x, y));
            }
        }
    }

    let calc_src_x = |dst_x: u32| -> u32 {
        let src_x = dst_x as i64 - left as i64;
        if src_x < 0 {
            // left
            src_x.unsigned_abs() as u32
        } else if (src_x as u32) < input_width {
            // middle
            src_x as u32
        } else {
            // right
            2 * input_width - src_x as u32 - 2
        }
    };
    // top
    for y in 0..top {
        let dst_y = y;
        let src_y = top - y;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output.unsafe_put_pixel(dst_x, dst_y, input.unsafe_get_pixel(src_x, src_y));
            }
        }
    }
    // bottom
    for y in 0..bottom {
        let dst_y = y + top + input_height;
        let src_y = input_height - y - 2;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output.unsafe_put_pixel(dst_x, dst_y, input.unsafe_get_pixel(src_x, src_y));
            }
        }
    }
    // middle
    for dst_y in top..(input_height + top) {
        let src_y = dst_y - top;
        for dst_x in 0..left {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output.unsafe_put_pixel(dst_x, dst_y, input.unsafe_get_pixel(src_x, src_y));
            }
        }
        for x in 0..right {
            let dst_x = x + left + input_width;
            let src_x = calc_src_x(dst_x);
            unsafe {
                output.unsafe_put_pixel(dst_x, dst_y, input.unsafe_get_pixel(src_x, src_y));
            }
        }
    }
    output
}

pub fn pad_array<A, S>(
    input: ArrayBase<S, Ix2>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
) -> Array2<A>
where
    S: ndarray::Data<Elem = A> + ndarray::RawData<Elem = A>,
    A: num_traits::Zero + Clone + Copy,
{
    let (input_width, input_height) = (input.ncols(), input.nrows());
    let (output_width, output_height) = (input_width + left + right, input_height + top + bottom);
    let mut output = ArrayBase::zeros((output_height, output_width));

    let output_ptr: *mut A = output.as_mut_ptr();
    let input_ptr = input.as_ptr();

    for y in 0..input_height {
        for x in 0..input_width {
            unsafe {
                output_ptr
                    .add(x + left + (y + top) * output_width)
                    .write(input_ptr.add(x + y * input_width).read());
                // output.unsafe_put_pixel(x + left, y + top, input.unsafe_get_pixel(x, y));
            }
        }
    }

    let calc_src_x = |dst_x: usize| -> usize {
        let src_x = dst_x as i64 - left as i64;
        if src_x < 0 {
            // left
            src_x.unsigned_abs() as usize
        } else if (src_x as usize) < input_width {
            // middle
            src_x as usize
        } else {
            // right
            2 * input_width - src_x as usize - 2
        }
    };
    // top
    for y in 0..top {
        let dst_y = y;
        let src_y = top - y;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    // bottom
    for y in 0..bottom {
        let dst_y = y + top + input_height;
        let src_y = input_height - y - 2;
        for dst_x in 0..output_width {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    // middle
    for dst_y in top..(input_height + top) {
        let src_y = dst_y - top;
        for dst_x in 0..left {
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
        for x in 0..right {
            let dst_x = x + left + input_width;
            let src_x = calc_src_x(dst_x);
            unsafe {
                output_ptr
                    .add(dst_x + dst_y * output_width)
                    .write(input_ptr.add(src_x + src_y * input_width).read());
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::sync::Once;

    use super::*;

    static INIT: Once = Once::new();
    fn setup() {
        INIT.call_once(|| {
            env_logger::init();
        });
    }

    #[test]
    fn test_calc_hist() {
        setup();
        // u8
        let input = imageproc::gray_image!(type: u8, 0,1,2; 2,4,2);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 2, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 3, 0, 1]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,2; 2,4,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 3, hist.as_mut_slice());
        let mut right = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![1, 1, 3, 0, 1].into_iter().enumerate() {
            right[i] = v;
        }
        right[256] = 2;
        right[258] = 1;

        assert_eq!(hist, right);
    }

    #[test]
    fn test_clip_hist() {
        setup();
        // u8
        let input = imageproc::gray_image!(type: u8, 0,1,2; 2,4,2; 2,5,2);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 3, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 5, 0, 1, 1]);
        clip_hist(hist.as_mut_slice(), 3);
        assert_eq!(hist, vec![2, 1, 3, 1, 1, 1]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,256; 256,4,256; 2,5,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 4, hist.as_mut_slice());
        let mut right = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![1, 1, 2, 0, 1, 1].into_iter().enumerate() {
            right[i] = v;
        }
        right[256] = 5;
        right[258] = 1;
        assert_eq!(hist, right);
        clip_hist(hist.as_mut_slice(), 3);
        let mut right = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![2, 1, 2, 0, 1, 1].into_iter().enumerate() {
            right[i] = v;
        }
        right[129] = 1; // redistributed residual
        right[256] = 3;
        right[258] = 1;
        assert_eq!(hist, right);
    }

    #[test]
    fn test_calc_lut() {
        setup();
        // u8
        let input = imageproc::gray_image!(type: u8, 0,1,2; 2,4,2);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 2, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 3, 0, 1]);
        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale as f32);
        // assert_eq!(lut, vec![43, 85, 213, 213, 255]); // for round half up
        assert_eq!(lut, vec![42, 85, 212, 212, 255]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,256; 256,4,256; 2,5,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_tile_hist(&input, 0, 0, 3, 4, hist.as_mut_slice());

        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale as f32);

        let mut right = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        // for (i, v) in vec![21, 43, 85, 85, 106, 128].into_iter().enumerate() { // for round half up
        for (i, v) in vec![21, 42, 85, 85, 106, 128].into_iter().enumerate() {
            right[i] = v;
        }
        for r in right.iter_mut().take(256).skip(6) {
            *r = 128;
        }
        right[256] = 234;
        right[257] = 234;
        right[258] = 255;
        assert_eq!(lut, right);
    }

    #[test]
    fn test_pad_u16() {
        setup();
        let input = imageproc::gray_image!(type: u16, 0,1,2; 3,4,5);
        let output = pad_image(&input, 1, 1, 1, 1);
        let expected =
            imageproc::gray_image!(type: u16, 4,3,4,5,4; 1,0,1,2,1; 4,3,4,5,4; 1,0,1,2,1);
        assert_eq!(output, expected);
    }

    #[test]
    fn test_pad_array() {
        setup();
        let input = array![[0u16, 1, 2], [3, 4, 5]];
        let output = pad_array(input, 1, 1, 1, 1);
        let expected = array![
            [4u16, 3, 4, 5, 4],
            [1, 0, 1, 2, 1],
            [4, 3, 4, 5, 4],
            [1, 0, 1, 2, 1]
        ];
        assert_eq!(output, expected);
    }

    fn _test_clahe_size(width: u32, height: u32, tile_sample: f64) {
        let input = image::GrayImage::new(width, height);
        println!("width, height, tile_sample: {width}, {height}, {tile_sample}");
        let output = clahe::<u8, u8>(&input, 8, 8, 8, tile_sample);
        assert_eq!(output.width(), input.width());
        assert_eq!(output.height(), input.height());
    }

    fn _test_clahe_size_smaple(tile_sample: f64) {
        _test_clahe_size(848, 1024, tile_sample);
        _test_clahe_size(848, 1020, tile_sample);
        _test_clahe_size(1234, 567, tile_sample);
        // maybe more random sizes
    }

    #[test]
    fn test_clahe_size_1() {
        setup();
        _test_clahe_size_smaple(1.0);
    }

    #[test]
    fn test_clahe_size_2() {
        setup();
        _test_clahe_size_smaple(2.0);
    }

    #[test]
    fn test_clahe_size_4() {
        setup();
        _test_clahe_size_smaple(4.0);
    }

    #[test]
    fn test_calculate_lut_and_weights() {
        let original_input_width = 8;
        let tile_width = 4;
        let tile_step_width = 4;
        let sampled_grid_width = 2;
        let lut_size = 1;
        let result = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        assert_eq!(
            result,
            vec![
                (0, 0, 0.5),
                (0, 0, 0.75),
                (0, 1, 0.0),
                (0, 1, 0.25),
                (0, 1, 0.5),
                (0, 1, 0.75),
                (1, 1, 0.0),
                (1, 1, 0.25),
            ]
        );

        let original_input_width = 8;
        let tile_width = 4;
        let tile_step_width = 2;
        let sampled_grid_width = 4;
        let lut_size = 1;
        let result = calculate_lut_and_weights(
            original_input_width,
            tile_width,
            tile_step_width,
            sampled_grid_width,
            lut_size,
        );
        assert_eq!(
            result,
            vec![
                (0, 0, 0.5),
                (0, 1, 0.0),
                (0, 1, 0.5),
                (1, 2, 0.0),
                (1, 2, 0.5),
                (2, 3, 0.0),
                (2, 3, 0.5),
                (3, 3, 0.0),
            ]
        );
    }
}

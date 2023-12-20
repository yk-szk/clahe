use image::*;
#[macro_use]
extern crate log;

type HistType = [u32];

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

fn clip_hist(hist: &mut HistType, limit: u32) {
    let mut clipped: u32 = 0;

    unsafe {
        for i in 0..hist.len() {
            let count = hist.get_unchecked_mut(i);
            if *count > limit {
                clipped += *count - limit;
                *count = limit;
            }
        }
    }

    if clipped > 0 {
        let hist_size = hist.len() as u32;
        let redist_batch = clipped / hist_size;

        if redist_batch > 0 {
            unsafe {
                for i in 0..hist.len() {
                    let count = hist.get_unchecked_mut(i);
                    *count += redist_batch;
                }
            }
        }

        let residual = (clipped - redist_batch * hist_size) as usize;
        assert!(residual < hist_size as usize);
        if residual > 0 {
            let step = usize::max(1, hist_size as usize / residual);
            let end = step * residual;
            for index in (0..end).step_by(step) {
                unsafe {
                    *hist.get_unchecked_mut(index) += 1;
                }
            }
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

fn calc_lut<T: CastFrom<T>>(hist: &HistType, lut: &mut [T], scale: f32) {
    let mut cumsum: u32 = 0;
    unsafe {
        for index in 0..hist.len() {
            cumsum += *hist.get_unchecked(index);
            *lut.get_unchecked_mut(index) = T::cast_from(round(cumsum as f32 * scale));
        }
    }
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE)
///
/// # Arguments
/// * `input` - GrayImage or Gray16Image
///
/// The CLAHE implementation below is based OpenCV which was licensed under Apache 2 License.
pub fn clahe<T, S>(
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
) -> Result<ImageBuffer<Luma<S>, Vec<S>>, Box<dyn std::error::Error>>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + CastFrom<T> + Default + 'static,
    S: image::Primitive + Into<usize> + Into<u32> + Ord + CastFrom<S> + 'static,
    f32: From<T> + From<S>,
{
    let (input_width, input_height) = input.dimensions();
    let mut output = ImageBuffer::<Luma<S>, Vec<S>>::new(input_width, input_height);

    debug!("Original size {} x {}", input_width, input_height);
    let input = if input_width % grid_width != 0 || input_height % grid_height != 0 {
        let pad_width = (grid_width - input_width % grid_width) % grid_width;
        let pad_height = (grid_height - input_height % grid_height) % grid_height;
        debug!("Padding image by w{} and h{}", pad_width, pad_height);
        pad_image(input, 0, pad_height, 0, pad_width)
    } else {
        input.clone() // TODO: avoid copying
    };

    let tile_width = input.dimensions().0 / grid_width;
    let tile_height = input.dimensions().1 / grid_height;
    debug!("Tile size {} x {}", tile_width, tile_height);
    let max_pix_value = *input.iter().max().unwrap();

    // max_pixe_value + 1 is used as the size of the histogram to reduce the computation for clip_hist ans calc_lut.
    // However, this is different from OpenCV's size (T::Max + 1).
    // While this difference does not affect test images in tests directories,
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

    debug!("Calculate lookup tables");
    let mut lookup_tables: Vec<T> =
        vec![T::default(); (grid_width * grid_height * lut_size) as usize];

    let mut hist = vec![0; hist_size as usize];
    unsafe {
        for slice_idx in 0..grid_height {
            let slice = &mut lookup_tables[(slice_idx * grid_width * lut_size) as usize
                ..((slice_idx + 1) * grid_width * lut_size) as usize];
            for row_idx in 0..grid_width {
                let lut =
                    &mut slice[(row_idx * lut_size) as usize..((row_idx + 1) * lut_size) as usize];

                let (left, top, width, height) = (
                    tile_width * row_idx,
                    tile_height * slice_idx,
                    tile_width,
                    tile_height,
                );

                calc_tile_hist(&input, left, top, width, height, hist.as_mut_slice());
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
        let mut lr_luts = vec![(0, 0); input_width as usize];
        let mut x_weights = vec![0.0; input_width as usize];
        for x in 0..(input_width as usize) {
            let left_x = (x as f64 / tile_width as f64 - 0.5).floor();
            let right_x = std::cmp::min((left_x + 1.0) as u32, grid_width - 1);
            let x_weight = (x as Float / tile_width as Float - 0.5 - left_x as Float) as Float;
            let left_x = left_x as u32;
            *lr_luts.get_unchecked_mut(x) = (left_x * lut_size, right_x * lut_size);
            *x_weights.get_unchecked_mut(x) = x_weight;
        }
        // perform interpolation
        for y in 0..(input_height as usize) {
            let top_y = (y as Float / tile_height as Float - 0.5).floor();
            let bottom_y = std::cmp::min((top_y + 1.0) as u32, grid_height - 1);
            let y_weight = (y as Float / tile_height as Float - 0.5 - top_y as Float) as Float;
            let top_y = top_y as u32; // -0.5f64 => 0u32
            let output_row_ptr = output_ptr.add(y * input_width as usize);
            let top_lut = &lookup_tables[(top_y * grid_width * lut_size) as usize
                ..((top_y + 1) * grid_width * lut_size) as usize];
            let bottom_lut = &lookup_tables[(bottom_y * grid_width * lut_size) as usize
                ..((bottom_y + 1) * grid_width * lut_size) as usize];
            for x in 0..(input_width as usize) {
                let input_pixel: u32 = input.unsafe_get_pixel(x as u32, y as u32).0[0].into();
                let x_weight = *x_weights.get_unchecked(x);
                let (left, right) = lr_luts.get_unchecked(x);

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

    Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calc_hist() {
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
        let input = imageproc::gray_image!(type: u16, 0,1,2; 3,4,5);
        let output = pad_image(&input, 1, 1, 1, 1);
        let expected =
            imageproc::gray_image!(type: u16, 4,3,4,5,4; 1,0,1,2,1; 4,3,4,5,4; 1,0,1,2,1);
        assert_eq!(output, expected);
    }
}

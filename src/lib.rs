use image::*;
#[macro_use]
extern crate log;

type HistType = [u32];
type LuTType = [u8];

fn calc_hist<I: GenericImageView>(img: &I, hist: &mut HistType)
where
    I::Pixel: 'static,
    <I::Pixel as Pixel>::Subpixel: Into<usize> + 'static,
{
    hist.fill(0);
    let hist_ptr = hist.as_mut_ptr();
    let (width, height) = img.dimensions();
    unsafe {
        for y in 0..height {
            for x in 0..width {
                let pix = img.unsafe_get_pixel(x, y);
                *hist_ptr.add((pix.channels()[0]).into()) += 1;
            }
        }
    }
}

fn clip_hist(hist: &mut HistType, limit: u32) {
    let mut clipped: u32 = 0;

    let hist_ptr = hist.as_mut_ptr();

    unsafe {
        for i in 0..hist.len() {
            let count = hist_ptr.add(i);
            if *count > limit {
                clipped += *count - limit;
                *count = limit;
            }
        }
    }

    let hist_size = hist.len() as u32;
    let redist_batch = clipped / hist_size;

    if redist_batch > 0 {
        unsafe {
            for i in 0..hist.len() {
                let count = hist_ptr.add(i);
                *count += redist_batch;
            }
        }
    }

    let mut residual = (clipped - redist_batch * hist_size) as usize;
    assert!(residual < hist_size as usize);
    if residual > 0 {
        let step = usize::max(1, (hist_size as usize / residual) as usize);
        for index in (0..hist_size as usize).step_by(step) {
            unsafe {
                *hist_ptr.add(index) += 1;
            }
            // histogram.channels[index] += 1;
            residual -= 1;
            if residual == 0 {
                break;
            }
        }
    }
}

fn calc_lut(hist: &HistType, lut: &mut LuTType, scale: f64) {
    let mut cumsum: u32 = 0;
    let hist_ptr = hist.as_ptr();
    let lut_ptr = lut.as_mut_ptr();
    unsafe {
        for index in 0..hist.len() {
            cumsum += *hist_ptr.add(index);
            *lut_ptr.add(index) = (cumsum as f64 * scale).round() as u8;
        }
    }
}

/// Contrast Limited Adaptive Histogram Equalization (CLAHE)
///
/// # Arguments
/// - `input` - GrayImage or Gray16Image
///
pub fn clahe<T>(
    input: &ImageBuffer<Luma<T>, Vec<T>>,
    grid_width: u32,
    grid_height: u32,
    clip_limit: u32,
) -> Result<GrayImage, Box<dyn std::error::Error>>
where
    T: image::Primitive + Into<usize> + Into<u32> + Ord + 'static,
{
    let (input_width, input_height) = input.dimensions();
    let mut output = GrayImage::new(input_width, input_height);

    debug!(
        "original width {}, original height{}",
        input_width, input_height
    );
    let input = if input_width % grid_width != 0 || input_height % grid_height != 0 {
        let pad_width = (grid_width - input_width % grid_width) % grid_width;
        let pad_height = (grid_height - input_height % grid_height) % grid_height;
        debug!("Padding image by w{} and h{}", pad_width, pad_height);
        pad_image(&input, 0, pad_height, 0, pad_width)
    } else {
        input.clone() // TODO: avoid copying
    };

    let tile_width = input.dimensions().0 / grid_height;
    let tile_height = input.dimensions().1 / grid_width;
    debug!("tile width {}, tile height {}", tile_width, tile_height);
    let max_pix_value = *input.iter().max().unwrap();
    let hist_size = usize::max(u8::MAX as usize, max_pix_value.into()) + 1;
    // let hist_size: usize = usize::max(u8::MAX as usize, max_pix_value.into()) + 1;
    let hist_size = if (hist_size-1) > u8::MAX as usize {u16::MAX as usize} else {u8::MAX as usize};
    let lut_size = hist_size as u32;
    let lut_scale = u8::MAX as f64 / (tile_width * tile_height) as f64;

    debug!("calculate lookup tables");
    let mut lookup_tables: Vec<u8> = vec![0; (grid_width * grid_height * lut_size) as usize];

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

    let mut hist = vec![0; hist_size as usize];
    let lut_ptr = lookup_tables.as_mut_ptr();
    unsafe {
        for slice_idx in 0..grid_width {
            let slice: &mut LuTType = &mut lookup_tables[(slice_idx * grid_height * lut_size)
                as usize
                ..((slice_idx + 1) * grid_height * lut_size) as usize];
            for row_idx in 0..grid_height {
                let lut: &mut LuTType =
                    &mut slice[(row_idx * lut_size) as usize..((row_idx + 1) * lut_size) as usize];
                let tile = input.view(
                    tile_width * row_idx as u32,
                    tile_height * slice_idx as u32,
                    tile_width,
                    tile_height,
                );

                calc_hist(&tile, hist.as_mut_slice());
                if clip_limit > 1 {
                    clip_hist(hist.as_mut_slice(), clip_limit);
                }
                calc_lut(hist.as_mut_slice(), lut, lut_scale);
            }
        }
        type FLOAT = f32;

        debug!("apply interpolations");
        let mut left_luts = vec![0; input_width as usize];
        let mut right_luts = vec![0; input_width as usize];
        let mut x_weights: Vec<FLOAT> = vec![0.0; input_width as usize];
        let output_ptr = output.as_mut_ptr();
        for x in 0..(output.dimensions().0 as usize) {
            let left_x = (x as f64 / tile_width as f64 - 0.5).floor();
            let right_x = std::cmp::min((left_x + 1.0) as u32, grid_height - 1);
            let x_weight = (x as FLOAT / tile_width as FLOAT - 0.5 - left_x as FLOAT) as FLOAT;
            let left_x = left_x as u32;
            left_luts[x] = left_x * lut_size;
            right_luts[x] = right_x * lut_size;
            x_weights[x] = x_weight;
        }
        let left_luts_ptr = left_luts.as_ptr();
        let right_luts_ptr = right_luts.as_ptr();
        let x_weights_ptr = x_weights.as_ptr();
        for y in 0..(output.dimensions().1 as usize) {
            let top_y = (y as FLOAT / tile_height as FLOAT - 0.5).floor();
            let bottom_y = std::cmp::min((top_y + 1.0) as u32, grid_width - 1);
            let y_weight = (y as FLOAT / tile_height as FLOAT - 0.5 - top_y as FLOAT) as FLOAT;
            let top_y = top_y as u32; // -0.5f64 => 0u32
            let output_row_ptr = output_ptr.add(y * input_width as usize);
            let top_lut_ptr = lut_ptr.add((top_y * grid_height * lut_size) as usize);
            let bottom_lut_ptr = lut_ptr.add((bottom_y * grid_height * lut_size) as usize);
            for x in 0..(output.dimensions().0 as usize) {
                let input_pixel: u32 = input.unsafe_get_pixel(x as u32, y as u32).0[0].into();
                let x_weight = *x_weights_ptr.add(x);

                let top_left = *top_lut_ptr.add((input_pixel + (*left_luts_ptr.add(x))) as usize);
                let bottom_left =
                    *bottom_lut_ptr.add((input_pixel + (*left_luts_ptr.add(x))) as usize);
                let top_right = *top_lut_ptr.add((input_pixel + *(right_luts_ptr.add(x))) as usize);
                let bottom_right =
                    *bottom_lut_ptr.add((input_pixel + *(right_luts_ptr.add(x))) as usize);

                #[inline]
                fn interpolate<T: Into<FLOAT>>(left: T, right: T, right_weight: FLOAT) -> FLOAT {
                    let left: FLOAT = left.into();
                    let right: FLOAT = right.into();
                    left as FLOAT * (1.0 - right_weight) + right as FLOAT * right_weight
                }
                let intermediate_1 = interpolate(top_left, top_right, x_weight);
                let intermediate_2 = interpolate(bottom_left, bottom_right, x_weight);
                let interpolated =
                    interpolate(intermediate_1, intermediate_2, y_weight).round() as u8;
                *output_row_ptr.add(x as usize) = interpolated;
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
            src_x.abs() as u32
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
        calc_hist(&input, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 3, 0, 1]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,2; 2,4,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_hist(&input, hist.as_mut_slice());
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
        calc_hist(&input, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 5, 0, 1, 1]);
        clip_hist(hist.as_mut_slice(), 3);
        assert_eq!(hist, vec![2, 1, 3, 1, 1, 1]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,256; 256,4,256; 2,5,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_hist(&input, hist.as_mut_slice());
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
        calc_hist(&input, hist.as_mut_slice());
        assert_eq!(hist, vec![1, 1, 3, 0, 1]);
        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale);
        assert_eq!(lut, vec![43, 85, 213, 213, 255]);

        // u16
        let input = imageproc::gray_image!(type: u16, 0,1,256; 256,4,256; 2,5,2; 256,258,256);
        let mut hist = vec![0u32; *input.iter().max().unwrap() as usize + 1];
        calc_hist(&input, hist.as_mut_slice());

        let mut lut = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        let scale: f64 = 255.0 / hist.iter().sum::<u32>() as f64;
        calc_lut(hist.as_slice(), lut.as_mut_slice(), scale);

        let mut right = vec![0u8; *input.iter().max().unwrap() as usize + 1];
        for (i, v) in vec![21, 43, 85, 85, 106, 128].into_iter().enumerate() {
            right[i] = v;
        }
        for i in 6..256 {
            right[i] = 128;
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

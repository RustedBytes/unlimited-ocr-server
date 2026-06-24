use anyhow::{Context, anyhow};
use image::{
    DynamicImage, GenericImage, ImageDecoder, ImageReader, Rgb, RgbImage, imageops::FilterType,
};
use log::trace;
use std::path::Path;

const GRAY_PAD: Rgb<u8> = Rgb([127, 127, 127]);
const EXACT_RESIZE_LIMIT: u32 = 640;

pub(super) fn decode_image_with_orientation(image_path: &Path) -> anyhow::Result<DynamicImage> {
    let reader = ImageReader::open(image_path)
        .with_context(|| format!("failed to open image {}", image_path.display()))?
        .with_guessed_format()
        .context("failed to guess image format")?;
    let mut decoder = reader
        .into_decoder()
        .context("failed to create image decoder")?;
    let orientation = decoder
        .orientation()
        .context("failed to read image orientation")?;
    let mut image = DynamicImage::from_decoder(decoder).context("failed to decode image")?;
    image.apply_orientation(orientation);
    Ok(image)
}

pub(super) fn preprocess_image(image: DynamicImage, image_size: u32) -> anyhow::Result<Vec<f32>> {
    if image_size == 0 {
        return Err(anyhow!("image_size must be greater than zero"));
    }

    trace!(
        "normalizing Unlimited-OCR image target_width={} target_height={}",
        image_size, image_size
    );

    let image = image.to_rgb8();
    let contained = if image_size <= EXACT_RESIZE_LIMIT {
        DynamicImage::ImageRgb8(image)
            .resize_exact(image_size, image_size, FilterType::CatmullRom)
            .to_rgb8()
    } else {
        resize_to_fit(&image, image_size)?
    };
    let padded = pad_to_square(&contained, image_size)?;
    Ok(normalize_chw(&padded))
}

fn resize_to_fit(image: &RgbImage, image_size: u32) -> anyhow::Result<RgbImage> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return Err(anyhow!("image dimensions must be greater than zero"));
    }

    let scale =
        (f64::from(image_size) / f64::from(width)).min(f64::from(image_size) / f64::from(height));
    let resized_width = ((f64::from(width) * scale).round() as u32).max(1);
    let resized_height = ((f64::from(height) * scale).round() as u32).max(1);

    Ok(DynamicImage::ImageRgb8(image.clone())
        .resize_exact(resized_width, resized_height, FilterType::CatmullRom)
        .to_rgb8())
}

fn pad_to_square(image: &RgbImage, image_size: u32) -> anyhow::Result<RgbImage> {
    let (width, height) = image.dimensions();
    if width > image_size || height > image_size {
        return Err(anyhow!(
            "resized image {width}x{height} does not fit target square {image_size}x{image_size}"
        ));
    }

    let mut output = RgbImage::from_pixel(image_size, image_size, GRAY_PAD);
    let x = (image_size - width) / 2;
    let y = (image_size - height) / 2;
    output
        .copy_from(image, x, y)
        .map_err(|err| anyhow!("failed to pad image: {err}"))?;
    Ok(output)
}

fn normalize_chw(image: &RgbImage) -> Vec<f32> {
    let (width, height) = image.dimensions();
    let plane = (width * height) as usize;
    let mut chw = vec![0.0_f32; 3 * plane];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = image.get_pixel(x as u32, y as u32).0;
            let dst = y * width as usize + x;
            for channel in 0..3 {
                let value = pixel[channel] as f32 / 255.0;
                chw[channel * plane + dst] = (value - 0.5) / 0.5;
            }
        }
    }

    chw
}

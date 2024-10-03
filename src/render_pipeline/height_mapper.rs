use image::{GenericImageView, Pixel};
use std::path::Path;

use bevy::{asset::LoadState, prelude::*};

#[derive(Default, Resource)]
pub struct WorldSetup;

// Define a resource to store the heightmap data
#[derive(Default, Resource)]
pub struct Heightmap(pub Vec<Vec<u32>>);
pub fn load_and_process_heightmap() -> Option<Heightmap> {
    let image_path = Path::new("assets/heightmapdata/Take1.png");

    match image::open(&image_path) {
        Ok(img) => {
            let (width, height) = img.dimensions();
            let sample_step = 1; // Adjust as needed for sampling resolution

            let sampled_width = (width as usize + sample_step - 1) / sample_step;
            let sampled_height = (height as usize + sample_step - 1) / sample_step;

            let mut heightmap = vec![vec![0u32; sampled_width]; sampled_height];

            for (sampled_y, y) in (0..height).step_by(sample_step).enumerate() {
                for (sampled_x, x) in (0..width).step_by(sample_step).enumerate() {
                    let pixel = img.get_pixel(x, y);

                    // Convert pixel to luminance (grayscale value)
                    let luminance = pixel.to_luma()[0] as u32;

                    // Store the luminance directly as the height value
                    heightmap[sampled_y][sampled_x] = luminance;
                }
            }

            println!("Heightmap loaded and processed successfully.");
            Some(Heightmap(heightmap))
        }
        Err(e) => {
            error!(
                "Failed to load heightmap image from {:?}: {}",
                image_path, e
            );
            None
        }
    }
}
pub fn run_setup(height_map: Option<Res<Heightmap>>, world_setup: Option<Res<WorldSetup>>) -> bool {
    if let Some(_) = world_setup {
        if let Some(_) = height_map {
            return true;
        }
    }
    false
}

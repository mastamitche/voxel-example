use bevy::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::height_mapper::Heightmap;
use super::{
    cpu_brickmap::{Brick, CpuBrickmap},
    BRICK_SIZE,
};
use crate::ultilities::timeit::{timeit, timeitmut};

const REGION_SIZE: usize = 512; // Blocks per region along one axis
const WORLD_ORIGIN_OFFSET: u32 = 4096; // Arbitrary offset to handle negative regions

/// Loads the palette from a JSON file.
fn load_palette() -> HashMap<String, [u8; 4]> {
    let file = std::fs::File::open("assets/palette/blockstates.json")
        .expect("Failed to open palette file");
    let mut json: HashMap<String, [u8; 4]> =
        serde_json::from_reader(file).expect("Failed to parse palette JSON");
    // Insert default or fallback colors
    json.insert("".to_string(), [200, 200, 200, 127]);
    json.insert("minecraft:grass".to_string(), [0, 0, 0, 0]);
    json.insert("minecraft:tall_grass".to_string(), [0, 0, 0, 0]);
    json.insert("minecraft:grass_block".to_string(), [62, 204, 18, 255]);
    json.insert("minecraft:water".to_string(), [20, 105, 201, 30]);
    json.insert("minecraft:cave_air".to_string(), [0, 0, 0, 0]);
    json.insert("minecraft:lava".to_string(), [255, 123, 0, 255]);
    json.insert("minecraft:seagrass".to_string(), [62, 204, 18, 255]);
    json.insert("minecraft:deepslate".to_string(), [77, 77, 77, 255]);
    json.insert("minecraft:oak_log".to_string(), [112, 62, 8, 255]);
    json.insert("minecraft:oak_stairs".to_string(), [112, 62, 8, 255]);

    json
}
pub fn setup_voxels(heightmap: Heightmap, world_depth: u32) -> CpuBrickmap {
    let heightmap_data = &heightmap.0;
    //println!("Heightmap data {:?}", heightmap_data);
    let height_scale: f32 = 200.0;
    let mut brickmap = CpuBrickmap::new(world_depth - BRICK_SIZE.trailing_zeros());

    let palette = load_palette();
    let grass_color = palette["minecraft:grass_block"];
    let dirt_color = palette["minecraft:dirt"];

    let chunk_side_length_bricks = 16 / BRICK_SIZE;
    let side_length = 1 << world_depth;

    for chunk_x in 0..side_length / 16 {
        for chunk_z in 0..side_length / 16 {
            for brick_y in 0..chunk_side_length_bricks {
                for brick_x in 0..chunk_side_length_bricks {
                    for brick_z in 0..chunk_side_length_bricks {
                        let mut brick = Brick::empty();
                        for x in 0..BRICK_SIZE {
                            for y in 0..BRICK_SIZE {
                                for z in 0..BRICK_SIZE {
                                    let global_x = chunk_x * 16 + brick_x * BRICK_SIZE + x;
                                    let global_y = brick_y * BRICK_SIZE + y;
                                    let global_z = chunk_z * 16 + brick_z * BRICK_SIZE + z;

                                    if global_x >= heightmap_data.len() as u32
                                        || global_z >= heightmap_data[0].len() as u32
                                    {
                                        continue;
                                    }
                                    println!(
                                        "Hieghtmap data x y {:?}, global x y {:?}",
                                        heightmap_data[global_z as usize][global_x as usize],
                                        (global_x, global_y)
                                    );
                                    let surface_y = (heightmap_data[global_z as usize]
                                        [global_x as usize]
                                        as f32
                                        * height_scale)
                                        as u32;

                                    if global_y <= surface_y {
                                        let color = if global_y == surface_y {
                                            grass_color
                                        } else {
                                            dirt_color
                                        };
                                        let pos = UVec3::new(x as u32, y as u32, z as u32);
                                        brick.write(pos, color);
                                    }
                                }
                            }
                        }

                        let brick_pos = UVec3::new(
                            chunk_x as u32 * chunk_side_length_bricks as u32 + brick_x as u32,
                            brick_y as u32,
                            chunk_z as u32 * chunk_side_length_bricks as u32 + brick_z as u32,
                        );

                        brickmap
                            .place_brick(brick, brick_pos)
                            .expect("Failed to place brick");
                    }
                }
            }
        }
    }

    brickmap.recreate_mipmaps();
    brickmap
}

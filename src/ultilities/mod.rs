use bevy::prelude::*;

use timeit::*;

pub mod timeit;

pub struct UtilsPlugin;

impl Plugin for UtilsPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<TimingAggregator>()
            .add_systems(Update, (update_timing_aggregator, log_timings).chain());
    }
}

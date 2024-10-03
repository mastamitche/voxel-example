use bevy::prelude::*;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

static TIMING_DATA: Lazy<Mutex<HashMap<String, Vec<Duration>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

pub struct TimingData {
    pub count: u32,
    pub total: Duration,
    pub max: Duration,
    pub min: Duration,
    pub values: Vec<Duration>,
}

#[derive(Resource)]
pub struct TimingAggregator {
    pub data: HashMap<String, TimingData>,
    pub last_log: SystemTime,
}

impl Default for TimingAggregator {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
            last_log: SystemTime::now(),
        }
    }
}

impl TimingAggregator {
    pub fn add_timing(&mut self, description: &str, duration: Duration) {
        let entry = self
            .data
            .entry(description.to_string())
            .or_insert(TimingData {
                count: 0,
                total: Duration::new(0, 0),
                max: Duration::new(0, 0),
                min: Duration::new(u64::MAX, 999_999_999),
                values: Vec::new(),
            });
        entry.count += 1;
        entry.total += duration;
        entry.max = entry.max.max(duration);
        entry.min = entry.min.min(duration);
        entry.values.push(duration);
    }
}

pub fn timeit<F: Fn() -> T, T>(description: &str, f: F) -> T {
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();

    TIMING_DATA
        .lock()
        .unwrap()
        .entry(description.to_string())
        .or_insert_with(Vec::new)
        .push(duration);

    result
}

pub fn timeitmut<F, T>(description: &str, mut f: F) -> T
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let result = f();
    let duration = start.elapsed();

    TIMING_DATA
        .lock()
        .unwrap()
        .entry(description.to_string())
        .or_insert_with(Vec::new)
        .push(duration);

    result
}

pub fn update_timing_aggregator(mut timing_aggregator: ResMut<TimingAggregator>) {
    let mut timing_data = TIMING_DATA.lock().unwrap();
    for (description, durations) in timing_data.drain() {
        for duration in durations {
            timing_aggregator.add_timing(&description, duration);
        }
    }
}

pub fn log_timings(mut timing_aggregator: ResMut<TimingAggregator>) {
    let now = SystemTime::now();
    if now.duration_since(timing_aggregator.last_log).unwrap() >= Duration::from_secs(1) {
        for (description, data) in &timing_aggregator.data {
            let avg = data.total / data.count;
            let median = if data.values.is_empty() {
                Duration::new(0, 0)
            } else {
                let mut sorted_values = data.values.clone();
                sorted_values.sort();
                sorted_values[sorted_values.len() / 2]
            };

            println!(
                "{}: Avg: {}, Max: {}, Min: {}, Median: {}",
                description,
                format_duration(avg),
                format_duration(data.max),
                format_duration(data.min),
                format_duration(median)
            );
        }

        timing_aggregator.data.clear();
        timing_aggregator.last_log = now;
        println!("=================================");
    }
}

fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs_f64();
    if total_secs < 0.000001 {
        format!("{} ns", duration.as_nanos())
    } else if total_secs < 0.001 {
        format!("{:.2} µs", duration.as_micros())
    } else if total_secs < 1.0 {
        format!("{:.2} ms", duration.as_millis())
    } else if total_secs < 60.0 {
        format!("{:.2} s", total_secs)
    } else if total_secs < 3600.0 {
        let minutes = (total_secs / 60.0).floor();
        let seconds = total_secs % 60.0;
        format!("{:.0}m {:.2}s", minutes, seconds)
    } else {
        let hours = (total_secs / 3600.0).floor();
        let minutes = ((total_secs % 3600.0) / 60.0).floor();
        let seconds = total_secs % 60.0;
        format!("{:.0}h {:.0}m {:.2}s", hours, minutes, seconds)
    }
}

use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use image::GenericImageView;
use ndarray::{Array1, Array4};
use rayon::prelude::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use smartstring::alias;

use crate::{
    args::Prepare,
    consts::{IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH},
};

pub fn run(data: PathBuf, args: Prepare) -> Result<()> {
    let observations = index_observations(data.join("plates"), data.join("plates/plates.csv"))?;
    println!("Building train.array");
    build_observation_array(data.join("train.array"), &observations.train)?;
    build_label_array(data.join("train.label.array"), &observations.train)?;
    println!("Building test.array");
    build_observation_array(data.join("test.array"), &observations.test)?;
    build_label_array(data.join("test.label.array"), &observations.test)?;
    println!("Building valid.array");
    build_observation_array(data.join("valid.array"), &observations.valid)?;
    build_label_array(data.join("valid.label.array"), &observations.valid)?;
    Ok(())
}

#[derive(Debug, Deserialize, Serialize)]
struct IndexEntry {
    #[serde(rename = "class id")]
    class_id: alias::String,
    filepaths: alias::String,
    labels: alias::String,
    #[serde(rename = "data set")]
    data_set: alias::String,
}

struct Observations {
    train: Vec<IndexEntry>,
    test: Vec<IndexEntry>,
    valid: Vec<IndexEntry>,
}

impl Display for Observations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Observations [ train: {}, test: {}, valid: {} ]",
            self.train.len(),
            self.test.len(),
            self.valid.len()
        )
    }
}

fn index_observations(data: impl AsRef<Path>, index: impl AsRef<Path>) -> Result<Observations> {
    let data = data.as_ref();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(File::open(index)?));

    let mut train = vec![];
    let mut test = vec![];
    let mut valid = vec![];

    for result in rdr.deserialize() {
        let mut record: IndexEntry = result?;
        record.filepaths = data
            .join(record.filepaths.as_str())
            .to_str()
            .unwrap()
            .into();
        match record.data_set.as_str() {
            "train" => train.push(record),
            "test" => test.push(record),
            "valid" => valid.push(record),
            e => return Err(anyhow!("unknown data set: {}", e)),
        }
    }

    Ok(Observations { train, test, valid })
}

fn build_observation_array(file: impl AsRef<Path>, entries: &[IndexEntry]) -> Result<()> {
    let mut observations =
        Array4::<f32>::zeros((entries.len(), IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH));
    observations
        .axis_iter_mut(ndarray::Axis(0))
        .zip(entries.iter())
        .map(anyhow::Ok)
        .par_bridge()
        .try_for_each(|res| -> Result<()> {
            let (mut v, e) = res.unwrap();
            let img = image::open(e.filepaths.as_str())?;
            for y in 0..img.height() {
                for x in 0..img.width() {
                    let pixel = img.get_pixel(x, y);
                    v[[0, y as usize, x as usize]] = pixel.0[0] as f32 / 255.;
                    v[[1, y as usize, x as usize]] = pixel.0[1] as f32 / 255.;
                    v[[2, y as usize, x as usize]] = pixel.0[2] as f32 / 255.;
                }
            }

            Ok(())
        })?;

    let mut file = BufWriter::new(File::create(file)?);
    rmp_serde::encode::write(&mut file, &observations)?;

    Ok(())
}

fn build_label_array(file: impl AsRef<Path>, entries: &[IndexEntry]) -> Result<()> {
    let label_map = entries
        .iter()
        .map(|v| v.labels.clone())
        .collect::<HashSet<alias::String>>()
        .into_iter()
        .enumerate()
        .map(|(idx, label)| (label, idx))
        .collect::<HashMap<alias::String, usize>>();

    println!("Labels Found: {}", label_map.len());

    let array = entries
        .iter()
        .map(|v| *label_map.get(&v.labels).unwrap() as i64)
        .collect::<Array1<_>>();

    let mut file = BufWriter::new(File::create(file)?);
    rmp_serde::encode::write(&mut file, &array)?;

    Ok(())
}

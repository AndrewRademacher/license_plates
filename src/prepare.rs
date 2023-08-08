use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Display,
    fs::File,
    io::{BufReader, BufWriter},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use image::GenericImageView;
use ndarray::{Array1, Array4, ArrayView, Dimension};
use rayon::prelude::{ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use smartstring::alias;
use spinoff::{spinners, Color, Spinner};

use crate::{
    args::Prepare,
    consts::{IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH},
};

pub fn run(data: PathBuf, args: Prepare) -> Result<()> {
    let observations = index_observations(data.join("plates"), data.join("plates/plates.csv"))?;

    let spinner = Spinner::new(spinners::Aesthetic, "Building label map...", Color::Cyan);
    let label_map = build_label_index(&observations.train);
    spinner.stop_with_message(&format!("Label map contains {} labels.", label_map.len()));
    for (name, number) in label_map.iter() {
        println!("\t{:2}: {}", number, name);
    }

    let spinner = Spinner::new(spinners::Aesthetic, "Building train arrays...", Color::Cyan);
    let norm = build_observation_array(data.join("train.array"), &observations.train, None)?;
    build_label_array(
        data.join("train.label.array"),
        &observations.train,
        &label_map,
    )?;
    spinner.stop_with_message("Train arrays complete.");

    let spinner = Spinner::new(spinners::Aesthetic, "Building test arrays...", Color::Cyan);
    build_observation_array(data.join("test.array"), &observations.test, Some(norm))?;
    build_label_array(
        data.join("test.label.array"),
        &observations.test,
        &label_map,
    )?;
    spinner.stop_with_message("Test arrays complete.");

    let spinner = Spinner::new(spinners::Aesthetic, "Building valid arrays...", Color::Cyan);
    build_observation_array(data.join("valid.array"), &observations.valid, Some(norm))?;
    build_label_array(
        data.join("valid.label.array"),
        &observations.valid,
        &label_map,
    )?;
    spinner.stop_with_message("Valid arrays complete.");

    std::fs::write(&args.norm, serde_json::to_vec_pretty(&norm)?)?;
    println!("Saved normalization to {:?}", args.norm);

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

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Normalization {
    mean: f32,
    sd: f32,
}

impl Normalization {
    pub fn new<D: Dimension>(view: ArrayView<f32, D>) -> Self {
        let mean = view.mean().unwrap();
        let sd = view.std(0.);
        Self { mean, sd }
    }

    #[inline]
    pub fn norm(&self, value: f32) -> f32 {
        (value - self.mean) / self.sd
    }
}

fn build_observation_array(
    file: impl AsRef<Path>,
    entries: &[IndexEntry],
    norm: Option<Normalization>,
) -> Result<Normalization> {
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

    let norm = match norm {
        Some(norm) => norm,
        None => Normalization::new(observations.view()),
    };
    observations.par_mapv_inplace(|v| norm.norm(v));

    let mut file = BufWriter::new(File::create(file)?);
    rmp_serde::encode::write(&mut file, &observations)?;

    Ok(norm)
}

fn build_label_index(entries: &[IndexEntry]) -> BTreeMap<alias::String, usize> {
    entries
        .iter()
        .map(|v| v.labels.clone())
        .collect::<BTreeSet<alias::String>>()
        .into_iter()
        .enumerate()
        .map(|(idx, label)| (label, idx))
        .collect::<BTreeMap<alias::String, usize>>()
}

fn build_label_array(
    file: impl AsRef<Path>,
    entries: &[IndexEntry],
    label_map: &BTreeMap<alias::String, usize>,
) -> Result<()> {
    let array = entries
        .iter()
        .map(|v| *label_map.get(&v.labels).unwrap() as i64)
        .collect::<Array1<_>>();

    let mut file = BufWriter::new(File::create(file)?);
    rmp_serde::encode::write(&mut file, &array)?;

    Ok(())
}

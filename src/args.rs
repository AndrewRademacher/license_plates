use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
pub struct Args {
    /// The data directory to use when preparing and training.
    #[clap(short = 'd', long = "data", about, default_value = "data")]
    pub data: PathBuf,
    #[clap(subcommand)]
    pub command: Command,
}

#[derive(Debug, Parser)]
pub enum Command {
    /// Translate the training set into an array structure ready for training.
    Prepare(Prepare),
    /// Train a model.
    Train(Train),
    /// Conduct an inference of one image with the trained model.
    Inference(Inference),
}

#[derive(Debug, Parser)]
pub struct Prepare {
    /// The file wherein normalization information is stored.
    #[clap(short = 'n', long = "norm", about)]
    pub norm: PathBuf,
    /// A mapping of string labels to integers.
    #[clap(short = 'l', long = "label", about)]
    pub label_map: PathBuf,
}

#[derive(Debug, Parser)]
pub struct Train {
    /// The location where the trained model will be saved.
    #[clap(short = 'm', long = "model", about)]
    pub model: PathBuf,
}

#[derive(Debug, Parser)]
pub struct Inference {
    /// The model to use when inferring.
    #[clap(short = 'm', long = "model", about)]
    pub model: PathBuf,
    /// The normalization values used in training.
    #[clap(short = 'n', long = "normalization", about)]
    pub normalization: PathBuf,
    /// The map of string labels to integers.
    #[clap(short = 'l', long = "label", about)]
    pub label_map: PathBuf,
    /// The image file to run inference upon.
    #[clap(short = 'i', long = "image", about)]
    pub image: Vec<PathBuf>,
}

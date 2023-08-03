use anyhow::Result;
use args::Args;
use clap::Parser;

mod args;
mod consts;
mod inference;
mod prepare;
mod train;

fn main() -> Result<()> {
    let args: Args = Args::parse();
    match args.command {
        args::Command::Prepare(sub) => prepare::run(args.data, sub),
        args::Command::Train(sub) => train::run(args.data, sub),
        args::Command::Inference(sub) => inference::run(sub),
    }
}

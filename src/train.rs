use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{Array1, Array4};
use spinoff::{spinners, Color, Spinner};
use tch::{
    nn::{self, ModuleT, OptimizerConfig},
    vision::{
        dataset::Dataset,
        resnet::{resnet152, resnet34, resnet50},
        vgg::vgg16,
    },
    Device, Tensor,
};

use crate::{
    args::Train,
    consts::LABLES,
    networks::{fast_resnet::fast_resnet, vgg::vgg_16},
};

pub fn run(data: PathBuf, _args: Train) -> Result<()> {
    // let device = Device::Mps;
    // let device = Device::Cpu;
    let device = Device::cuda_if_available();

    let spinner = Spinner::new(spinners::Aesthetic, "Loading training set...", Color::Cyan);
    let m = load_dataset(data)?;
    spinner.stop_with_message("Training set loaded.");

    let spinner = Spinner::new(spinners::Aesthetic, "Constructing network...", Color::Cyan);
    let vs = nn::VarStore::new(device);
    // let net = fast_resnet(&vs.root());
    // let net = vgg_16(&vs.root(), LABLES, true);
    let net = resnet50(&vs.root(), LABLES);
    spinner.stop_with_message("Network constructed.");

    let epochs = 10i64;
    let batch_size = 16;
    let mut test_accuracy = 0.;
    let progress = ProgressBar::new(m.train_iter(batch_size).count() as u64 * epochs as u64);
    progress.set_style(
        ProgressStyle::with_template("[{eta_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
            .unwrap(),
    );

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    // let mut opt = nn::AdamW::default().build(&vs, 1e-3)?;
    for epoch in 0..epochs {
        opt.set_lr(learning_rate(epoch));
        for (bimages, blabels) in m.train_iter(batch_size).shuffle().to_device(vs.device()) {
            // let bimages = tch::vision::dataset::augmentation(&bimages, true, 4, 8);
            let loss = net
                .forward_t(&bimages, true)
                // .cross_entropy_loss(
                //     &blabels,
                //     None::<Tensor>,
                //     tch::Reduction::Mean,
                //     -100,
                //     0.0,
                // );
                .cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);

            progress.set_message(format!(
                "{:4.4} | {:3.2}%",
                loss.double_value(&[]),
                100. * test_accuracy
            ));
            progress.inc(1);
        }

        test_accuracy = progress.suspend(|| {
            let spinner = Spinner::new(spinners::Aesthetic, "Testing accuracy...", Color::Cyan);
            let value =
                net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 64);
            spinner.stop();
            value
        });
    }
    progress.finish();
    Ok(())
}

fn learning_rate(epoch: i64) -> f64 {
    if epoch < 1 {
        0.001
    } else if epoch < 2 {
        0.0001
    } else if epoch < 3 {
        0.00001
    } else {
        0.000001
    }
}

fn load_dataset(data: impl AsRef<Path>) -> Result<Dataset> {
    let data = data.as_ref();
    Ok(Dataset {
        train_images: read_image_tensor(data.join("train.array"))?,
        train_labels: read_label_tensor(data.join("train.label.array"))?,
        test_images: read_image_tensor(data.join("test.array"))?,
        test_labels: read_label_tensor(data.join("test.label.array"))?,
        labels: LABLES,
    })
}

fn read_image_tensor(file: impl AsRef<Path>) -> Result<Tensor> {
    let array: Array4<f32> = rmp_serde::decode::from_read(BufReader::new(File::open(file)?))?;
    let tensor = array.try_into()?;
    Ok(tensor)
}

fn read_label_tensor(file: impl AsRef<Path>) -> Result<Tensor> {
    let array: Array1<i64> = rmp_serde::decode::from_read(BufReader::new(File::open(file)?))?;
    let tensor = array.try_into()?;
    Ok(tensor)
}

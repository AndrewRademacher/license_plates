use std::{collections::BTreeMap, time::Instant};

use anyhow::Result;
use image::GenericImageView;
use ndarray::Array4;
use tch::{
    nn::{self, ModuleT},
    vision::resnet::resnet50,
    Device, Tensor,
};

use crate::{
    args::Inference,
    consts::{IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, LABELS},
    prepare::{LabelMap, Normalization},
};

pub fn run(args: Inference) -> Result<()> {
    let device = Device::cuda_if_available();

    let mut vs = nn::VarStore::new(device);
    let net = resnet50(&vs.root(), LABELS);
    vs.load(args.model)?;

    let norm: Normalization = serde_json::from_slice(&std::fs::read(args.normalization)?)?;
    let label_map: LabelMap = serde_json::from_slice(&std::fs::read(args.label_map)?)?;
    let int_label_map = label_map
        .iter()
        .map(|(k, v)| (*v, k.clone()))
        .collect::<BTreeMap<_, _>>();

    for (idx, img_path) in args.image.into_iter().enumerate() {
        let timer = Instant::now();
        let img = image::open(img_path)?;
        let mut array = Array4::<f32>::zeros((1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH));
        for y in 0..img.height() {
            for x in 0..img.width() {
                let pixel = img.get_pixel(x, y);
                array[[0, 0, y as usize, x as usize]] = pixel.0[0] as f32 / 255.;
                array[[0, 1, y as usize, x as usize]] = pixel.0[1] as f32 / 255.;
                array[[0, 2, y as usize, x as usize]] = pixel.0[2] as f32 / 255.;
            }
        }
        drop(img);
        array.par_mapv_inplace(|v| norm.norm(v));
        let tensor: Tensor = array.try_into()?;
        let tensor = tensor.to_device(device);

        let out = net.forward_t(&tensor, false).softmax(-1, tch::Kind::Float);
        let out = out.argmax(1, false).int64_value(&[0]);
        println!(
            "Plate {} is from {} ({}s)",
            idx,
            int_label_map.get(&(out as usize)).unwrap(),
            timer.elapsed().as_secs_f32()
        );
    }
    Ok(())
}

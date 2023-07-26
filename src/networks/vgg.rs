use tch::nn::{self, ConvConfig, SequentialT};

pub fn vgg_16(vs: &nn::Path, classes: i64, train: bool) -> SequentialT {
    nn::seq_t()
        .add(conv_block(vs, 3, 64))
        .add(conv_pool_block(vs, 64, 64))
        .add(conv_block(vs, 64, 128))
        .add(conv_pool_block(vs, 128, 128))
        .add(conv_block(vs, 128, 256))
        .add(conv_block(vs, 256, 256))
        .add(conv_pool_block(vs, 256, 256))
        .add(conv_block(vs, 256, 512))
        .add(conv_block(vs, 512, 512))
        .add(conv_pool_block(vs, 512, 512))
        .add_fn(|x| x.flat_view())
        .add(linear_block(vs, 57344, 4096, train))
        .add(linear_block(vs, 4096, 4096, train))
        .add(nn::linear(vs, 4096, classes, Default::default()))
}

fn conv_block(vs: &nn::Path, i: i64, o: i64) -> SequentialT {
    let conv_config = ConvConfig {
        padding: 1,
        stride: 1,
        ..Default::default()
    };

    nn::seq_t()
        .add(nn::conv2d(vs, i, o, 3, conv_config))
        .add(nn::batch_norm2d(vs, o, Default::default()))
        .add_fn(|x| x.relu())
}

fn conv_pool_block(vs: &nn::Path, i: i64, o: i64) -> SequentialT {
    let conv_config = ConvConfig {
        padding: 1,
        stride: 1,
        ..Default::default()
    };

    nn::seq_t()
        .add(nn::conv2d(vs, i, o, 3, conv_config))
        .add(nn::batch_norm2d(vs, o, Default::default()))
        .add_fn(|x| x.relu())
        .add_fn(|x| x.max_pool2d_default(2))
}

fn linear_block(vs: &nn::Path, i: i64, o: i64, train: bool) -> SequentialT {
    nn::seq_t()
        .add_fn(move |x| x.dropout(0.5, train))
        .add(nn::linear(vs, i, o, Default::default()))
        .add_fn(|x| x.relu())
}

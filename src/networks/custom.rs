use tch::{
    nn::{self, ConvConfig},
    Tensor,
};

use crate::consts::{IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, LABLES};

#[derive(Debug)]
struct Net {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    conv3: nn::Conv2D,
    conv4: nn::Conv2D,
    conv5: nn::Conv2D,
    conv6: nn::Conv2D,
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        let conv_config = ConvConfig {
            padding: 1,
            ..Default::default()
        };
        Self {
            conv1: nn::conv2d(vs, 3, 64, 5, conv_config),
            conv2: nn::conv2d(vs, 64, 64, 5, conv_config),
            conv3: nn::conv2d(vs, 64, 128, 5, conv_config),
            conv4: nn::conv2d(vs, 128, 128, 5, conv_config),
            conv5: nn::conv2d(vs, 128, 256, 5, conv_config),
            conv6: nn::conv2d(vs, 256, 256, 5, conv_config),
            fc1: nn::linear(vs, 65536, 512, Default::default()),
            fc2: nn::linear(vs, 512, LABLES, Default::default()),
        }
    }
}

impl nn::ModuleT for Net {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.view([
            -1,
            IMAGE_CHANNELS as i64,
            IMAGE_HEIGHT as i64,
            IMAGE_WIDTH as i64,
        ])
        .apply(&self.conv1)
        .relu()
        .apply(&self.conv2)
        .relu()
        .max_pool2d_default(4)
        .apply(&self.conv3)
        .relu()
        .apply(&self.conv4)
        .relu()
        .max_pool2d_default(4)
        .apply(&self.conv5)
        .relu()
        .apply(&self.conv6)
        .relu()
        .max_pool2d_default(2)
        .view([-1])
        .apply(&self.fc1)
        .relu()
        .dropout(0.5, train)
        .apply(&self.fc2)
    }
}

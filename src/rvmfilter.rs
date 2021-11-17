use crate::filter::{Filter, FilterError};
use once_cell::sync::Lazy;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::IntoDimension;
use onnxruntime::ndarray::{self, Dimension, IxDyn};
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use opencv::core::{Scalar, Size};
use opencv::dnn::blob_from_image;
use opencv::prelude::*;
use std::mem;
use std::path::Path;

#[derive(Debug)]
pub struct RVMFilter<'a> {
    session: Session<'a>,
    downsample_ratio: ndarray::ArrayD<f32>,
    r1o: ndarray::ArrayD<f32>,
    r2o: ndarray::ArrayD<f32>,
    r3o: ndarray::ArrayD<f32>,
    r4o: ndarray::ArrayD<f32>,
}

static ORT_ENV: Lazy<Environment> = Lazy::new(|| {
    Environment::builder()
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .with_name("rvmruntime")
        .build()
        .expect("Failed to build ORT environment")
});

const DOWNSAMPLE_RATIO: f32 = 0.5;

impl<'a> RVMFilter<'a> {
    pub fn new<P: AsRef<Path> + 'a>(model_file: P) -> Result<RVMFilter<'a>, FilterError> {
        let session = (&*ORT_ENV)
            .new_session_builder()?
            .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)?
            .with_number_threads(4)?
            .with_model_from_file(model_file)?;
        // Initialize recurrent values with 1x1x1x1 all-zero array
        let recurrent_init = ndarray::ArrayD::<f32>::zeros(IxDyn(&[1, 1, 1, 1]));

        Ok(RVMFilter {
            session,
            downsample_ratio: ndarray::ArrayD::from_elem(IxDyn(&[1]), DOWNSAMPLE_RATIO),
            r1o: recurrent_init.clone(),
            r2o: recurrent_init.clone(),
            r3o: recurrent_init.clone(),
            r4o: recurrent_init,
        })
    }
}

fn tensor2mat<D>(tensor: OrtOwnedTensor<f32, D>) -> Result<opencv::core::Mat, FilterError>
where
    D: Dimension,
{
    let tensor_dims: Vec<i32> = (*tensor)
        .dim()
        .into_dimension()
        .as_array_view()
        .iter()
        .map(|dim| *dim as i32)
        .collect();
    let mat = opencv::core::Mat::from_slice((&*tensor).as_slice().ok_or_else(|| {
        FilterError::Other(String::from(
            "Failed to get output tensor array view as slice",
        ))
    })?)?;
    Ok(mat.reshape_nd(1, tensor_dims.as_slice())?)
}

impl<'a> Filter for RVMFilter<'a> {
    fn filter_inplace(&mut self, src_image: &mut Mat, bg_image: &Mat) -> Result<(), FilterError> {
        // Ensure that we have two HWC images with three channels
        if src_image.dims() != 2 || src_image.channels().unwrap() != 3 {
            return Err(FilterError::Other(format!(
                "Expected a WHC source image (where C=3), got {:?} with {} channels",
                src_image.mat_size(),
                src_image.channels().unwrap()
            )));
        }
        if bg_image.dims() != 2 || bg_image.channels().unwrap() != 3 {
            return Err(FilterError::Other(format!(
                "Expected a WHC background image (where C=3), got {:?} with {} channels",
                bg_image.mat_size(),
                bg_image.channels().unwrap()
            )));
        }
        // Ensure that the images have the same dimensions
        if *(src_image.mat_size()) != *(bg_image.mat_size()) {
            return Err(FilterError::Other(format!(
                "Camera image has size {:?} but background image has size {:?}.",
                src_image.mat_size(),
                bg_image.mat_size()
            )));
        }

        let blob = blob_from_image(
            src_image,
            1.0,                                           //Don't scale
            Size::new(src_image.cols(), src_image.rows()), //Use original image size
            Scalar::new(0.0, 0.0, 0.0, 0.0),               // Add nothing
            false,                                         // Don't swap R and B
            false,                                         // Don't crop
            opencv::core::CV_32F,                          // Use f32 values
        )?;

        let bs = blob.mat_size();
        let src_shape = ndarray::IxDyn(&[
            bs.get(0)? as usize,
            bs.get(1)? as usize,
            bs.get(2)? as usize,
            bs.get(3)? as usize,
        ]);
        let downsample_ratio = 1.0;

        let inputs: Vec<ndarray::ArrayD<f32>> = vec![
            unsafe {
                ndarray::ArrayViewD::<f32>::from_shape_ptr(src_shape, blob.data() as *const f32)
                    .to_owned()
            },
            // We use take to replace the recurrent values with a default value and then moves
            // them out (to the session) in order to save us some copies.
            mem::take(&mut self.r1o),
            mem::take(&mut self.r2o),
            mem::take(&mut self.r3o),
            mem::take(&mut self.r4o),
            ndarray::ArrayD::<f32>::from_elem(IxDyn(&[1]), downsample_ratio),
        ];

        let outputs = self.session.run::<f32, f32, ndarray::IxDyn>(inputs)?;

        match outputs.as_slice() {
            [fgr, pha, r1o, r2o, r3o, r4o] => {
                // Set recurrent states to the ones we obtained
                self.r1o = (*r1o).to_owned();
                self.r2o = (*r2o).to_owned();
                self.r3o = (*r3o).to_owned();
                self.r4o = (*r4o).to_owned();
                Ok(())
            }
            other => Err(FilterError::Other(format!(
                "Expected six output tensors, got {}",
                other.len()
            ))),
        }
    }
}

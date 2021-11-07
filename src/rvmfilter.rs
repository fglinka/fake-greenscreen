use opencv::prelude::*;
use opencv::dnn::blob_from_image;
use opencv::core::{Size, Scalar};
use crate::filter::{Filter, FilterError};
use onnxruntime::environment::Environment;
use onnxruntime::session::Session;
use std::path::Path;
use once_cell::sync::Lazy;
use onnxruntime::ndarray;

#[derive(Debug)]
pub struct RVMFilter<'a> {
    session: Session<'a>
}

static ORT_ENV: Lazy<Environment> = Lazy::new(| | {
    Environment::builder()
        .with_log_level(onnxruntime::LoggingLevel::Verbose)
        .with_name("rvmruntime")
        .build().expect("Failed to build ORT environment")
});

impl<'a> RVMFilter<'a> {
    pub fn new<P: AsRef<Path> + 'a>(model_file: P) -> Result<RVMFilter<'a>, FilterError> {
        let session = (&*ORT_ENV).new_session_builder()?
                    .with_optimization_level(onnxruntime::GraphOptimizationLevel::Basic)?
                    .with_number_threads(4)?
                    .with_model_from_file(model_file)?;

        Ok(RVMFilter {
            session
        })
    }
}


impl<'a> Filter for RVMFilter<'a> {
    fn filter_inplace(&mut self, src_image: &mut Mat, bg_image: &Mat) -> Result<(), FilterError> {
        let src_size = src_image.mat_size();
        let bg_size = bg_image.mat_size();
        // Ensure that we have two HWC images with three channels which have the same size
        //if src_size.dims() != 3 || src_size.get(2).unwrap() != 3 {
            //return Err(FilterError::Other(format!("Expected a WHC camera image (where C=3), got {:?}", src_size)))
        //}
        //if bg_image.dims() != 3 || bg_size.get(2).unwrap() != 3 {
            //return Err(FilterError::Other(format!("Expected a WHC background image (where C=3), got {:?}", bg_size)))
        //}
        if *src_size != *bg_size {
            return Err(FilterError::Other(format!("Camera image has size {:?} but background image has size {:?}.",
                src_size, bg_size)));
        }

        let blob = blob_from_image(src_image,
            1.0, //Don't scale
            Size::new(src_image.cols(), src_image.rows()), //Use original image size
            Scalar::new(0.0, 0.0, 0.0, 0.0), // Add nothing
            false, // Don't swap R and B
            false, // Don't crop
            opencv::core::CV_32F // Use f32 values
        )?;

        let bs = blob.mat_size();
        let src_shape = ndarray::IxDyn(&[bs.get(0)? as usize, bs.get(1)? as usize, bs.get(2)? as usize, bs.get(3)? as usize]);
        let downsample_ratio = 1.0;

        let inputs: Vec<ndarray::ArrayD<f32>> = vec![
            unsafe { ndarray::ArrayViewD::<f32>::from_shape_ptr(src_shape, blob.data() as *const f32).to_owned() },
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 1, 1])),
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 1, 1])),
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 1, 1])),
            ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[1, 1, 1, 1])),
            ndarray::ArrayD::<f32>::from_elem(ndarray::IxDyn(&[1]), downsample_ratio)
        ];
        let outputs = self.session.run::<f32, f32, ndarray::IxDyn>(inputs)?;

        println!("Outputs: {:?}", outputs);

        Ok(())
    }
}

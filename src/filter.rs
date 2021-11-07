use opencv::prelude::*;
use quick_error::quick_error;

quick_error!{
    #[derive(Debug)]
    pub enum FilterError {
        Other(description: String) {
            display("Filter failed: {}", description)
        }
        CvError(err: opencv::Error) {
            display("Filter failed; OpenCV error: {}", err)
            from()
        }
        #[cfg(feature = "rvm")]
        OnnxError(err: onnxruntime::OrtError) {
            display("Filter failed, ONNXRuntime error: {}", err)
            from()
        }
    }
}

pub trait Filter {
    fn filter_inplace(&mut self, src_image: &mut Mat, bg_image: &Mat) -> Result<(), FilterError>;

    fn filter(&mut self, src_image: &Mat, bg_image: &Mat) -> Result<Mat, FilterError> {
        let mut mod_image = src_image.clone();
        self.filter_inplace(&mut mod_image, bg_image)?;
        Ok(mod_image)
    }
}

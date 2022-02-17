//! This filter does absolutely nothing to the data and is used
//! as a fallback by the gstreamer plugin if no filter was configured
//! or is available

use crate::filter::{Filter, FilterError};
use opencv::prelude::*;

#[derive(Debug)]
pub struct NoopFilter {}

impl Default for NoopFilter {
    fn default() -> Self {
        NoopFilter {}
    }
}

impl Filter for NoopFilter {
    fn filter_inplace(&mut self, src_image: &mut Mat, bg_image: &Mat) -> Result<(), FilterError> {
        Ok(())
    }
}

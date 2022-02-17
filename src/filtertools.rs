//! This module implements some general-purpose image-processing functions that can be used
//! by filters.

use opencv::prelude::*;

/// Replace the pixels in `src` for which `fgr_mask` indicates background exists with pixels
/// from `bg`. The `fgr_mask` is expected to be a single-channel Mat with f32 values. It should
/// be 1 for foreground and 0 for background or in between. Further, `src`, `bg` and `fgr_mask`
/// have to have the same dimensions.
pub fn apply_mask(src: &Mat, bg: &Mat, fgr_mask: &Mat) -> Result<Mat, opencv::Error> {
    let mut mask_bc = Mat::new_rows_cols_with_default(src.rows(), src.cols(), opencv::core::CV_32FC3, opencv::core::Scalar::from((0.0, 0.0, 0.0)))?;
    opencv::core::mix_channels(
        fgr_mask,
        &mut mask_bc,
        &[0,0, 0,1, 0,2]
    )?;
    opencv::highgui::imshow("preview", &mask_bc)?;
    opencv::highgui::wait_key(15)?;
    let mask_bc_inv = opencv::core::Scalar::from((1.0, 1.0, 1.0)) - &mask_bc;
    //Ok((src * &mask_bc + bg * mask_bc_inv).into_result()?.to_mat()?)
    Ok(Mat::default())
}

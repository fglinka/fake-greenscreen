use crate::filter::Filter;
use crate::filter::FilterError;
use crate::noopfilter::NoopFilter;
use crate::rvmfilter::RVMFilter;
use core::ffi::c_void;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer::subclass::ElementMetadata;
use gstreamer::Caps;
use gstreamer::FlowError;
use gstreamer_base::subclass::base_transform::BaseTransformMode;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video::{VideoFormat, VideoFrameRef, VideoInfo};
use once_cell::sync::Lazy;
use opencv::core::{Scalar, CV_8UC3};
use opencv::prelude::*;
use std::sync::Mutex;

static FILTER_ERROR_CAT: Lazy<gstreamer::DebugCategory> = Lazy::new(|| {
    gstreamer::DebugCategory::new(
        "fakecam",
        gstreamer::DebugColorFlags::empty() | gstreamer::DebugColorFlags::FG_RED,
        Some("Fake green-screen plugin error"),
    )
});

impl Into<gstreamer::LoggableError> for FilterError {
    fn into(self) -> gstreamer::LoggableError {
        gstreamer::loggable_error!(&*FILTER_ERROR_CAT, format!("Filter error: {}", self))
    }
}

mod imp {
    use super::*;

    #[derive(Debug)]
    pub struct FakecamTransform {
        video_info: Mutex<VideoInfo>,
        filter: Mutex<Box<dyn Filter>>,
        bg_frame: Mutex<Mat>,
    }

    const GREEN: Lazy<Scalar> = Lazy::new(|| Scalar::new(0.0, 255.0, 0.0, 255.0));

    impl Default for FakecamTransform {
        fn default() -> Self {
            let fmt = VideoFormat::Rgb;
            let width: u32 = 1920;
            let height: u32 = 1080;
            FakecamTransform {
                video_info: Mutex::new(
                    VideoInfo::builder(fmt, width, height)
                        .build()
                        .expect("Default video info for transform was invalid."),
                ),
                //filter: Mutex::new(Box::new(NoopFilter::default())),
                filter: Mutex::new(Box::new(
                    RVMFilter::new("/home/xeef/Projekte/fakecam2/models/rvm_mobilenetv3_fp32.onnx")
                        .unwrap(),
                )),
                bg_frame: Mutex::new(
                    Mat::new_rows_cols_with_default(height as i32, width as i32, CV_8UC3, *GREEN)
                        .expect("Failed to create default background"),
                ),
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for FakecamTransform {
        const NAME: &'static str = "FakecamTransform";
        type Type = super::FakecamTransform;
        type ParentType = gstreamer_base::BaseTransform;
    }

    impl ObjectImpl for FakecamTransform {}

    impl GstObjectImpl for FakecamTransform {}

    impl ElementImpl for FakecamTransform {
        fn metadata() -> Option<&'static ElementMetadata> {
            static METADATA: Lazy<ElementMetadata> = Lazy::new(|| {
                ElementMetadata::new(
                    "Fakecam transform",
                    "Filter/Video",
                    "Replaces the background behind a person without a greenscreen",
                    "Felix Glinka <devglinka@posteo.eu>",
                )
            });

            Some(&*METADATA)
        }

        fn pad_templates() -> &'static [gstreamer::PadTemplate] {
            static TEMPLATES: Lazy<Vec<gstreamer::PadTemplate>> = Lazy::new(|| {
                let caps = Caps::new_simple(
                    "video/x-raw",
                    &[
                        (
                            "format",
                            &gstreamer::List::new(&[&gstreamer_video::VideoFormat::Rgb.to_str()]),
                        ),
                        ("interlace-mode", &"progressive"),
                    ],
                );

                let src_template = gstreamer::PadTemplate::new(
                    "src",
                    gstreamer::PadDirection::Src,
                    gstreamer::PadPresence::Always,
                    &caps,
                )
                .unwrap();

                let sink_template = gstreamer::PadTemplate::new(
                    "sink",
                    gstreamer::PadDirection::Sink,
                    gstreamer::PadPresence::Always,
                    &caps,
                )
                .unwrap();

                vec![src_template, sink_template]
            });

            TEMPLATES.as_ref()
        }
    }

    impl BaseTransformImpl for FakecamTransform {
        const MODE: BaseTransformMode = BaseTransformMode::AlwaysInPlace;
        const PASSTHROUGH_ON_SAME_CAPS: bool = false;
        const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

        fn set_caps(
            &self,
            incaps: &Caps,
            outcaps: &Caps,
        ) -> Result<(), gstreamer::LoggableError> {
            let mut info = self.video_info.lock().unwrap();
            let info_in = VideoInfo::from_caps(incaps)?;
            let info_out = VideoInfo::from_caps(outcaps)?;
            if info_in != info_out {
                return Err(gstreamer::loggable_error!(
                    &*FILTER_ERROR_CAT,
                    format!(
                        "Input and output caps different. Input {:?}, Output {:?}",
                        info_in, info_out
                    )
                ));
            } else {
                *info = info_in.clone();
            }
            // Resize green frame to new size
            let mut bg_frame = self.bg_frame.lock().unwrap();
            bg_frame
                .resize_with_default(info_in.height() as usize, *GREEN)
                .unwrap();
            bg_frame.set_cols(info_in.width() as i32);

            Ok(())
        }
        fn transform_ip(
            &self,
            buf: &mut gstreamer::BufferRef,
        ) -> Result<gstreamer::FlowSuccess, FlowError> {
            // Obtain lock on video info
            let info = self.video_info.lock().or_else(|e| {
                gstreamer::error!(&*FILTER_ERROR_CAT, "Failed to obtain mutex lock");
                Err(FlowError::Error)
            })?;
            // Read out buffer as gstreamer-video VideoFrame
            let mut frame = VideoFrameRef::from_buffer_ref_writable(buf, &*info).or_else(|e| {
                gstreamer::error!(&*FILTER_ERROR_CAT, "Failed to extract video frame: {}", e);
                Err(FlowError::Error)
            })?;
            if frame.n_planes() != 1 {
                gstreamer::error!(
                    &*FILTER_ERROR_CAT,
                    "Extracted frame has {} planes, should have 1",
                    frame.n_planes()
                );
                return Err(FlowError::Error);
            }
            // Obtain mutable pointer to the single plane containing the image data
            let frame_data_ptr: *mut u8 = frame
                .plane_data_mut(0)
                .map(|data| data.as_mut_ptr())
                .or_else(|e| {
                    gstreamer::error!(
                        &*FILTER_ERROR_CAT,
                        "Failed to read out frame plane: {}",
                        e
                    );
                    Err(FlowError::Error)
                })?;
            let mut frame_mat = unsafe {
                opencv::core::Mat::new_rows_cols_with_data(
                    frame.height() as i32,
                    frame.width() as i32,
                    opencv::core::CV_8UC3,
                    frame_data_ptr as *mut c_void,
                    opencv::core::Mat_AUTO_STEP,
                )
            }
            .or_else(|e| {
                gstreamer::error!(&*FILTER_ERROR_CAT, "Failed to decode as CV Mat: {}", e);
                Err(FlowError::Error)
            })?;

            {
                let mut filter = self.filter.lock().or_else(|e| {
                    gstreamer::error!(
                        &*FILTER_ERROR_CAT,
                        "Failed to obtain filter lock: {}",
                        e
                    );
                    Err(FlowError::Error)
                })?;
                let bg = self.bg_frame.lock().or_else(|e| {
                    gstreamer::error!(
                        &*FILTER_ERROR_CAT,
                        "Failed to obtain BG frame lock: {}",
                        e
                    );
                    Err(FlowError::Error)
                })?;

                (*filter)
                    .filter_inplace(&mut frame_mat, &*bg)
                    .or_else(|e| {
                        println!("Filter failed: {}", e);
                        gstreamer::error!(&*FILTER_ERROR_CAT, "Filtering failed: {}", e);
                        Err(FlowError::Error)
                    })?;
            }

            Ok(gstreamer::FlowSuccess::Ok)
        }
    }
}

glib::wrapper! {
    pub struct FakecamTransform(ObjectSubclass<imp::FakecamTransform>) @extends gstreamer_base::BaseTransform, gstreamer::Element, gstreamer::Object;
}

fn plugin_init(plugin: &gstreamer::Plugin) -> Result<(), glib::BoolError> {
    gstreamer::Element::register(
        Some(plugin),
        "fakecam",
        gstreamer::Rank::None,
        FakecamTransform::static_type(),
    )
}

gstreamer::plugin_define!(
    fakecamplugin,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    "2021-10-12"
);

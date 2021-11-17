use gstreamer::prelude::*;
use gtk::prelude::*;
use gtk::{Application, ApplicationWindow};
use gtk4 as gtk;
use crate::filter::Filter;

mod plugin;
mod filter;
#[cfg(feature = "rvm")]
mod rvmfilter;

fn main() {
    gstreamer::init().unwrap();
    plugin::plugin_register_static().unwrap();

    #[cfg(feature = "rvm")]
    {
        let mut filter = rvmfilter::RVMFilter::new("/home/xeef/Projekte/fakecam2/models/rvm_mobilenetv3_fp32.onnx").unwrap();
        let mut testMat = opencv::core::Mat::new_rows_cols_with_default(480, 640, opencv::core::CV_8UC3, opencv::core::Scalar::from((0.0, 0.0, 0.0))).unwrap();
        let bgMat = opencv::core::Mat::new_rows_cols_with_default(480, 640, opencv::core::CV_8UC3, opencv::core::Scalar::from((0.0, 0.0, 0.0))).unwrap();
        filter.filter_inplace(&mut testMat, &bgMat).unwrap();
    }

    let src = gstreamer::ElementFactory::make("v4l2src", Some("src")).unwrap();
    let cvt1 = gstreamer::ElementFactory::make("videoconvert", Some("cvt1")).unwrap();
    let filter = gstreamer::ElementFactory::make("fakecam", Some("filter")).unwrap();
    let cvt2 = gstreamer::ElementFactory::make("videoconvert", Some("cvt2")).unwrap();
    let sink = gstreamer::ElementFactory::make("v4l2sink", Some("sink")).unwrap();

    let pipeline = gstreamer::Pipeline::new(Some("testpipeline"));
    pipeline
        .add_many(&[&src, &cvt1, &filter, &cvt2, &sink])
        .unwrap();

    src.set_property_from_str("device", "/dev/video0");
    sink.set_property_from_str("device", "/dev/video4");

    src.link(&cvt1).unwrap();
    cvt1.link(&filter).unwrap();
    filter.link(&cvt2).unwrap();
    cvt2.link(&sink).unwrap();

    pipeline.set_state(gstreamer::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        use gstreamer::MessageView;

        match msg.view() {
            MessageView::Error(err) => {
                eprintln!(
                    "Error received from element {:?}: {}",
                    err.src().map(|s| s.path_string()),
                    err.error()
                );
                eprintln!("Debugging information: {:?}", err.debug());
                break;
            }
            MessageView::Eos(..) => break,
            _ => (),
        }
    }

    pipeline
        .set_state(gstreamer::State::Null)
        .expect("Unable to set the pipeline to the `Null` state");
}
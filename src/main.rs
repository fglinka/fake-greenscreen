use gstreamer::prelude::*;
use gtk::prelude::*;
use gtk::{Application, ApplicationWindow};
use gtk4 as gtk;

mod plugin;

fn main() {
    gstreamer::init().unwrap();
    plugin::plugin_register_static().unwrap();

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
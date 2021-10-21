use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer::subclass::prelude::*;
use gstreamer::subclass::ElementMetadata;
use gstreamer::Caps;
use gstreamer_base::subclass::base_transform::BaseTransformMode;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video::prelude::*;
use once_cell::sync::Lazy;

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct FakecamTransform {}

    #[glib::object_subclass]
    impl ObjectSubclass for FakecamTransform {
        const NAME: &'static str = "FakecamTransform";
        type Type = super::FakecamTransform;
        type ParentType = gstreamer_base::BaseTransform;
    }

    impl ObjectImpl for FakecamTransform {}

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

        fn transform_ip(
            &self,
            _element: &Self::Type,
            buf: &mut gstreamer::BufferRef,
        ) -> Result<gstreamer::FlowSuccess, gstreamer::FlowError> {
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

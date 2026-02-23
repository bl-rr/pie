pub use crate::api::inferlet::actor;
pub use crate::api::inferlet::adapter;
pub use crate::api::inferlet::core::classify;
pub use crate::api::inferlet::core::common::{
    Blob, BlobResult, DebugQueryResult, Model, Priority, Queue, SynchronizationResult,
    allocate_resources, deallocate_resources, export_resources, get_all_exported_resources,
    import_resources, release_exported_resources,
};
pub use crate::api::inferlet::core::forward;
pub use crate::api::inferlet::core::kvs;
pub use crate::api::inferlet::core::message;
pub use crate::api::inferlet::core::runtime;
pub use crate::api::inferlet::core::tokenize;
pub use crate::api::inferlet::image;
pub use crate::api::inferlet::zo;

wit_bindgen::generate!({
    path: "wit",
    world: "imports",
    with: {
         "wasi:io/poll@0.2.4": wasi::io::poll,
    },
    generate_all,
});

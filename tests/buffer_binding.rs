mod common;

use embree3::{BufferSource, BufferUsage, Error, Format, GeometryKind};

fn as_bytes<T>(s: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, std::mem::size_of_val(s)) }
}

#[test]
fn shared_vertex_buffer_reads_from_pre_sliced_offset() {
    // The unit triangle's 3 vertices sit at element offset K of a
    // host array; we bind `&bytes[K*12..]` as the shared VERTEX buffer. The new API
    // passes byteOffset = 0 (caller pre-slices), so embree reads the triangle. The
    // old double-offset bug read from 2*K*12 and missed.
    //
    // `verts` is declared *before* `scene`/`tri` so it outlives them (the new
    // shared binding ties the geometry's lifetime to the host data).
    let device = common::device();
    const K: usize = 4;
    let mut verts: Vec<[f32; 3]> = vec![[9.0; 3]; K]; // leading padding
    verts.extend_from_slice(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    verts.extend_from_slice(&[[0.0; 3]; 2]); // trailing pad for the 16-byte SSE tail
    let bytes = as_bytes(&verts);

    let mut scene = device.create_scene().unwrap();
    let mut tri = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    tri.set_shared_buffer(
        BufferUsage::VERTEX,
        0,
        Format::FLOAT3,
        &bytes[K * 12..],
        12,
        3,
    )
    .unwrap();
    tri.set_new_buffer::<[u32; 3]>(BufferUsage::INDEX, 0, Format::UINT3, 12, 1)
        .unwrap()
        .copy_from_slice(&[[0, 1, 2]]);
    let tri = tri.commit();
    scene.attach_geometry(&tri);
    scene.commit();

    assert!(
        common::cast_center_ray(&scene).hit.is_valid(),
        "shared vertex buffer must be read from the pre-sliced offset, proving the new API passes \
         byteOffset=0 and embree doesn't double-offset"
    );
}

#[test]
fn validation_rejects_unsound_shared_bindings() {
    let device = common::device();
    // Declared before `geom` so the (rejected) bindings even type-check
    // lifetime-wise.
    let tight = vec![[0.0f32; 3]; 8]; // 96 bytes
    let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();

    // A VERTEX buffer needs the last element readable to a 16-byte boundary -> req
    // 100 bytes, but `tight` is only 96. This is the latent OOB the PDF warned
    // about.
    assert_eq!(
        geom.set_shared_buffer(
            BufferUsage::VERTEX,
            0,
            Format::FLOAT3,
            as_bytes(&tight),
            12,
            8
        ),
        Err(Error::INVALID_ARGUMENT)
    );
    // count == 0.
    assert_eq!(
        geom.set_shared_buffer(
            BufferUsage::VERTEX,
            0,
            Format::FLOAT3,
            as_bytes(&tight),
            12,
            0
        ),
        Err(Error::INVALID_ARGUMENT)
    );
    // stride (8) < element size (12).
    assert_eq!(
        geom.set_shared_buffer(
            BufferUsage::VERTEX,
            0,
            Format::FLOAT3,
            as_bytes(&tight),
            8,
            4
        ),
        Err(Error::INVALID_ARGUMENT)
    );
    // UNDEFINED format.
    assert_eq!(
        geom.set_shared_buffer(
            BufferUsage::VERTEX,
            0,
            Format::UNDEFINED,
            as_bytes(&tight),
            12,
            4
        ),
        Err(Error::INVALID_ARGUMENT)
    );
}

#[test]
fn new_buffer_round_trips_and_get_buffer_reports_local() {
    let device = common::device();
    let mut geom = device.create_geometry(GeometryKind::TRIANGLE).unwrap();
    geom.set_new_buffer::<[f32; 3]>(BufferUsage::VERTEX, 0, Format::FLOAT3, 12, 4)
        .unwrap()
        .copy_from_slice(&[[1.0; 3], [2.0; 3], [3.0; 3], [4.0; 3]]);

    match geom.get_buffer(BufferUsage::VERTEX, 0) {
        Some(BufferSource::Local { layout, .. }) => {
            assert_eq!(layout.count, 4);
            assert_eq!(layout.stride, 12);
            assert_eq!(layout.format, Format::FLOAT3);
        }
        other => panic!("expected a Local buffer source, got {other:?}"),
    }
}

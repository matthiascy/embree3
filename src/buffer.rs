use crate::Error;
use std::{
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    ops::{Bound, Deref, DerefMut, RangeBounds},
};

use crate::{device::Device, sys::*, Format};

/// Non-zero integer type used to describe the size of a buffer.
pub type BufferSize = NonZeroUsize;

impl RTCFormat {
    /// Size in bytes of one element of this format, or `None` for
    /// [`Format::UNDEFINED`](crate::Format::UNDEFINED) (so a bogus format can
    /// never silently pass a size check).
    pub fn byte_size(self) -> Option<usize> {
        use RTCFormat::*;
        let n = match self {
            UNDEFINED => return None,
            GRID => return Some(mem::size_of::<RTCGrid>()),
            UCHAR | CHAR => 1,
            UCHAR2 | CHAR2 | USHORT | SHORT => 2,
            UCHAR3 | CHAR3 => 3,
            UCHAR4 | CHAR4 | USHORT2 | SHORT2 | UINT | INT | FLOAT => 4,
            USHORT3 | SHORT3 => 6,
            USHORT4 | SHORT4 | UINT2 | INT2 | FLOAT2 | ULLONG | LLONG => 8,
            UINT3 | INT3 | FLOAT3 => 12,
            UINT4
            | INT4
            | FLOAT4
            | ULLONG2
            | LLONG2
            | FLOAT2X2_ROW_MAJOR
            | FLOAT2X2_COLUMN_MAJOR => 16,
            FLOAT5 => 20,
            FLOAT6
            | ULLONG3
            | LLONG3
            | FLOAT2X3_ROW_MAJOR
            | FLOAT2X3_COLUMN_MAJOR
            | FLOAT3X2_ROW_MAJOR
            | FLOAT3X2_COLUMN_MAJOR => 24,
            FLOAT7 => 28,
            FLOAT8
            | ULLONG4
            | LLONG4
            | FLOAT2X4_ROW_MAJOR
            | FLOAT2X4_COLUMN_MAJOR
            | FLOAT4X2_ROW_MAJOR
            | FLOAT4X2_COLUMN_MAJOR => 32,
            FLOAT9 | FLOAT3X3_ROW_MAJOR | FLOAT3X3_COLUMN_MAJOR => 36,
            FLOAT10 => 40,
            FLOAT11 => 44,
            FLOAT12
            | FLOAT3X4_ROW_MAJOR
            | FLOAT3X4_COLUMN_MAJOR
            | FLOAT4X3_ROW_MAJOR
            | FLOAT4X3_COLUMN_MAJOR => 48,
            FLOAT13 => 52,
            FLOAT14 => 56,
            FLOAT15 => 60,
            FLOAT16 | FLOAT4X4_ROW_MAJOR | FLOAT4X4_COLUMN_MAJOR => 64,
        };
        Some(n)
    }
}

/// Types that may safely alias the raw bytes of an embree buffer when mapping
/// it as `[Self]`.
///
/// # Safety
///
/// Implementing this asserts ALL of:
/// - **Any bit pattern is valid:** every sequence of `size_of::<Self>()` bytes
///   is a valid `Self` which rules out niche types (`bool`, `char`,
///   non-`#[repr(C)]` enums, `NonZero*`, references, `Box`).
/// - **No pointers/provenance:** the bytes come from embree and carry no valid
///   Rust provenance, so `Self` must contain no references or raw pointers.
/// - **No interior mutability:** mapping hands out `&[Self]`;
///   `Cell`/`UnsafeCell`/ atomics would allow unsound mutation through an
///   aliased shared view.
/// - **Padding-agnostic:** correctness must not depend on the contents of
///   padding bytes.
///
/// For a custom vertex/index type: make it `#[repr(C)]` (or
/// `#[repr(transparent)]`), give it only `BufferData` fields, derive `Copy`,
/// then `unsafe impl BufferData for MyVertex {}`.
pub unsafe trait BufferData: Copy {}

// Fixed-width plain-data scalars. `usize`/`isize` are intentionally excluded
// (their width is platform-dependent).
unsafe impl BufferData for u8 {}
unsafe impl BufferData for i8 {}
unsafe impl BufferData for u16 {}
unsafe impl BufferData for i16 {}
unsafe impl BufferData for u32 {}
unsafe impl BufferData for i32 {}
unsafe impl BufferData for u64 {}
unsafe impl BufferData for i64 {}
unsafe impl BufferData for f32 {}
unsafe impl BufferData for f64 {}
unsafe impl<T: BufferData, const N: usize> BufferData for [T; N] {}

/// Layout of a geometry buffer binding: how embree reads the bound bytes.
/// Carried by every binding and reported by
/// [`Geometry::get_buffer`](crate::Geometry::get_buffer).
#[derive(Debug, Clone, Copy)]
pub struct BufferLayout {
    /// Element format of the bound data.
    pub format: Format,
    /// Byte stride between consecutive elements.
    pub stride: usize,
    /// Number of elements.
    pub count: usize,
}

/// The data source bound to a geometry buffer slot.
///
/// The *only* place the buffer provenance is a sum type (binding uses the typed
/// `set_*_buffer` methods).
#[derive(Debug)]
pub enum BufferSource<'a> {
    /// A sub-range of a refcounted [`Buffer`]. Owns a retained clone (a
    /// reference would dangle once the geometry's internal lock guard
    /// drops).
    Managed {
        /// The refcounted buffer (a retained clone).
        buffer: Buffer,
        /// Byte offset of the bound sub-range within `buffer`.
        byte_offset: usize,
        /// How embree reads the bound bytes.
        layout: BufferLayout,
    },
    /// Caller-owned host memory shared zero-copy with embree.
    Shared {
        /// The caller-owned host bytes shared with embree.
        data: &'a [u8],
        /// How embree reads the bound bytes.
        layout: BufferLayout,
    },
    /// Embree-owned, geometry-local storage. Opaque (no raw pointer exposed);
    /// read it via the geometry's mapping methods.
    Local {
        /// Size of the embree-owned storage.
        size: BufferSize,
        /// How embree reads the bound bytes.
        layout: BufferLayout,
    },
}

/// Bytes embree may read for `count` (>= 1) elements of `format`, each `stride`
/// apart. A **vertex** buffer (`VERTEX`/`VERTEX_ATTRIBUTE`) SSE-reads the last
/// element up to a 16-byte boundary, so its tail is `round_up(elem, 16)`.
/// Returns `None` on an unknown format, `count == 0`, `stride` too small / not
/// 4-aligned, or overflow.
pub(crate) fn required_layout_bytes(
    format: Format,
    stride: usize,
    count: usize,
    vertex: bool,
) -> Option<usize> {
    if count == 0 {
        return None;
    }
    let elem = format.byte_size()?;
    if stride < elem || stride % 4 != 0 {
        return None;
    }
    let tail = if vertex {
        elem.checked_next_multiple_of(16)?
    } else {
        elem
    };
    (count - 1).checked_mul(stride)?.checked_add(tail)
}

#[cfg(test)]
mod format_tests {
    use crate::Format;

    #[test]
    fn format_byte_size() {
        assert_eq!(Format::FLOAT3.byte_size(), Some(12));
        assert_eq!(Format::UINT3.byte_size(), Some(12));
        assert_eq!(Format::FLOAT.byte_size(), Some(4));
        assert_eq!(Format::FLOAT4X4_COLUMN_MAJOR.byte_size(), Some(64));
        assert_eq!(Format::FLOAT2X3_ROW_MAJOR.byte_size(), Some(24));
        assert_eq!(Format::UCHAR4.byte_size(), Some(4));
        assert_eq!(Format::ULLONG2.byte_size(), Some(16));
        assert_eq!(Format::UNDEFINED.byte_size(), None);
        assert_eq!(
            Format::GRID.byte_size(),
            Some(std::mem::size_of::<crate::sys::RTCGrid>())
        );
    }

    #[test]
    fn required_layout_bytes_rules() {
        use super::required_layout_bytes as req;
        // FLOAT3, 8 elements, stride 12.
        assert_eq!(req(Format::FLOAT3, 12, 8, false), Some(96)); // 7*12 + 12
        assert_eq!(req(Format::FLOAT3, 12, 8, true), Some(100)); // vertex tail -> +16
        assert_eq!(req(Format::FLOAT4, 16, 8, true), Some(128)); // already 16-aligned
                                                                 // Rejections.
        assert_eq!(req(Format::FLOAT3, 12, 0, false), None); // count == 0
        assert_eq!(req(Format::UNDEFINED, 12, 8, false), None); // unknown format
        assert_eq!(req(Format::FLOAT3, 8, 8, false), None); // stride < elem (12)
        assert_eq!(req(Format::FLOAT3, 13, 8, false), None); // stride not 4-aligned
        assert_eq!(req(Format::FLOAT3, usize::MAX, 8, false), None); // overflow
    }
}

/// Handle to a buffer managed by Embree.
#[derive(Debug)]
pub struct Buffer {
    pub(crate) device: Device,
    pub(crate) handle: RTCBuffer,
    pub(crate) size: BufferSize,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        unsafe { rtcRetainBuffer(self.handle) };
        Buffer {
            device: self.device.clone(),
            handle: self.handle,
            size: self.size,
        }
    }
}

impl Buffer {
    /// Creates a new data buffer of the given size.
    pub(crate) fn new(device: &Device, size: BufferSize) -> Result<Buffer, Error> {
        // Pad to a multiple of 16 bytes
        let size = if size.get() % 16 == 0 {
            size.get()
        } else {
            (size.get() + 15) & !15
        };
        let handle = unsafe { rtcNewBuffer(device.handle, size) };
        if handle.is_null() {
            Err(device.get_error())
        } else {
            Ok(Buffer {
                handle,
                size: NonZeroUsize::new(size).unwrap(),
                device: device.clone(),
            })
        }
    }

    /// Returns the raw handle to the buffer.
    ///
    /// # Safety
    ///
    /// Use this function only if you know what you are doing. The returned
    /// handle is a raw pointer to an Embree reference-counted object. The
    /// reference count is not increased by this function, so the caller must
    /// ensure that the handle is not used after the buffer object is
    /// destroyed.
    pub unsafe fn handle(&self) -> RTCBuffer { self.handle }

    /// Slices into the buffer for the given range.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The range of indices to slice into the buffer.
    ///   - Ranges with no end will slice to the end of the buffer.
    ///   - Totally unbounded range (..) will slice the entire buffer.
    pub fn mapped_range<S: RangeBounds<usize>, T>(&self, bounds: S) -> BufferView<'_, T> {
        let (offset, size) = range_bounds_to_offset_and_size(bounds);
        let size = size.unwrap_or_else(|| self.size.get() - offset);
        debug_assert!(offset + size <= self.size.get() && offset < self.size.get());
        BufferView::new(self, offset, BufferSize::new(size).unwrap()).unwrap()
    }

    /// Mutable slice into the buffer for the given range.
    ///
    /// # Arguments
    ///
    /// * `bounds` - The range of indices to slice into the buffer.
    ///  - Ranges with no end will slice to the end of the buffer.
    /// - Totally unbounded range (..) will slice the entire buffer.
    pub fn mapped_range_mut<S: RangeBounds<usize>, T>(
        &mut self,
        bounds: S,
    ) -> BufferViewMut<'_, T> {
        let (offset, size) = range_bounds_to_offset_and_size(bounds);
        let size = size.unwrap_or_else(|| self.size.get() - offset);
        debug_assert!(offset + size <= self.size.get() && offset < self.size.get());
        BufferViewMut::new(self, offset, BufferSize::new(size).unwrap()).unwrap()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            rtcReleaseBuffer(self.handle);
        }
    }
}

/// A read-only view into mapped buffer.
#[derive(Debug)]
pub struct BufferView<'buf, T: 'buf> {
    mapped: BufferMappedRange<'buf, T>,
    marker: PhantomData<&'buf T>,
}

/// A write-only view into mapped buffer.
#[derive(Debug)]
pub struct BufferViewMut<'buf, T: 'buf> {
    mapped: BufferMappedRange<'buf, T>,
    marker: PhantomData<&'buf mut T>,
}

impl<'src, T> BufferView<'src, T> {
    /// Creates a new slice from the given Buffer with the given offset and
    /// size. Only used internally by [`Buffer::mapped_range`].
    fn new(
        buffer: &'src Buffer,
        offset: usize,
        size: BufferSize,
    ) -> Result<BufferView<'src, T>, Error> {
        Ok(BufferView {
            mapped: BufferMappedRange::from_buffer(buffer, offset, size.into())?,
            marker: PhantomData,
        })
    }
}

impl<'src, T> BufferViewMut<'src, T> {
    /// Creates a new slice from the given Buffer with the given offset and
    /// size. Only used internally by [`Buffer::mapped_range_mut`].
    fn new(
        buffer: &'src Buffer,
        offset: usize,
        size: BufferSize,
    ) -> Result<BufferViewMut<'src, T>, Error> {
        Ok(BufferViewMut {
            mapped: BufferMappedRange::from_buffer(buffer, offset, size.into())?,
            marker: PhantomData,
        })
    }
}

/// A slice of a mapped [`Buffer`].
#[derive(Debug)]
struct BufferMappedRange<'a, T: 'a> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<&'a mut T>, // covariant without drop check
}

impl<'a, T: 'a> BufferMappedRange<'a, T> {
    /// Creates a new slice from the given Buffer with the given offset and
    /// size.
    ///
    /// The offset and size must be in bytes and must be a multiple of the size
    /// of `T`.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the given offset and size are valid.
    fn from_buffer(
        buffer: &'a Buffer,
        offset: usize,
        size: usize,
    ) -> Result<BufferMappedRange<'a, T>, Error> {
        debug_assert!(
            size % mem::size_of::<T>() == 0,
            "Size of the range of the mapped buffer must be multiple of T!"
        );
        debug_assert!(
            offset % mem::size_of::<T>() == 0,
            "Offset must be multiple of T!"
        );
        let ptr = unsafe {
            let ptr = rtcGetBufferData(buffer.handle) as *const u8;
            if ptr.is_null() {
                return Err(buffer.device.get_error());
            }
            ptr.add(offset)
        } as *mut T;
        Ok(BufferMappedRange {
            ptr,
            len: size / mem::size_of::<T>(),
            marker: PhantomData,
        })
    }

    /// Creates a new slice from the given raw pointer and length.
    unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> BufferMappedRange<'a, T> {
        BufferMappedRange {
            ptr,
            len,
            marker: PhantomData,
        }
    }

    fn as_slice(&self) -> &[T] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<'buf, T: BufferData> BufferViewMut<'buf, T> {
    /// Builds a mutable view over `len` elements at `ptr` (e.g.
    /// embree-allocated geometry-local storage returned by
    /// `rtcSetNewGeometryBuffer`).
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned for `T`, and valid for `len` `T`s for
    /// `'buf`, and uniquely borrowed for that span (the caller holds `&'buf
    /// mut` of the owner).
    pub(crate) unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        BufferViewMut {
            mapped: BufferMappedRange::from_raw_parts(ptr, len),
            marker: PhantomData,
        }
    }
}

impl<'buf, T: BufferData> BufferView<'buf, T> {
    /// Builds a read-only view over `len` elements at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned for `T`, and valid for `len` `T`s for
    /// `'buf` (shared borrow; concurrent read views may coexist).
    pub(crate) unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        BufferView {
            mapped: BufferMappedRange::from_raw_parts(ptr, len),
            marker: PhantomData,
        }
    }
}

impl<'src, T> AsRef<[T]> for BufferView<'src, T> {
    fn as_ref(&self) -> &[T] { self.mapped.as_slice() }
}

impl<'src, T> AsMut<[T]> for BufferViewMut<'src, T> {
    fn as_mut(&mut self) -> &mut [T] { self.mapped.as_mut_slice() }
}

impl<'src, T> Deref for BufferView<'src, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { self.mapped.as_slice() }
}

impl<'src, T> Deref for BufferViewMut<'src, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target { self.mapped.as_slice() }
}

impl<'src, T> DerefMut for BufferViewMut<'src, T> {
    fn deref_mut(&mut self) -> &mut Self::Target { self.mapped.as_mut_slice() }
}

/// Converts a range bounds into an offset and size.
fn range_bounds_to_offset_and_size<S: RangeBounds<usize>>(bounds: S) -> (usize, Option<usize>) {
    let offset = match bounds.start_bound() {
        Bound::Included(&n) => n,
        Bound::Excluded(&n) => n + 1,
        Bound::Unbounded => 0,
    };
    let size = match bounds.end_bound() {
        Bound::Included(&n) => Some(n - offset + 1),
        Bound::Excluded(&n) => Some(n - offset),
        Bound::Unbounded => None,
    };

    (offset, size)
}

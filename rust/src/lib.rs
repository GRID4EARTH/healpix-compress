//! Native kernels of the `hcc-sentinel2-pyramid` codec of healpix-compress.
//!
//! The JPEG XL and zlib streams stay with imagecodecs and zlib on the Python
//! side; this module does the rest of `encode` and `decode` (z-order, parent
//! means, prediction, residual, masks) without holding the GIL.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

mod pyramid;

/// Version of the `HS2P` codestream these kernels implement.
const PYRAMID_FORMAT: u8 = 1;

impl From<pyramid::Error> for PyErr {
    fn from(e: pyramid::Error) -> PyErr {
        PyValueError::new_err(e.0)
    }
}

fn shape2<T: numpy::Element>(a: &PyReadonlyArray2<'_, T>) -> (usize, usize) {
    let s = a.shape();
    (s[0], s[1])
}

/// Decode the parts of a codestream to the nested chunk (float64, or float32 if `single`).
#[pyfunction]
#[pyo3(signature = (base_dn, r16, coef, k, offset, step, q, o, mask, single))]
#[allow(clippy::too_many_arguments)]
fn pyramid_decode<'py>(
    py: Python<'py>,
    base_dn: PyReadonlyArray2<'py, u16>,
    r16: PyReadonlyArray2<'py, u16>,
    coef: PyReadonlyArray2<'py, f32>,
    k: usize,
    offset: i64,
    step: f64,
    q: f64,
    o: f64,
    mask: Option<Bound<'py, PyBytes>>,
    single: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (h, w) = shape2(&base_dn);
    if h != w {
        return Err(PyValueError::new_err("the base must be square"));
    }
    let (b, r, c) = (base_dn.as_slice()?, r16.as_slice()?, coef.as_slice()?);
    let m = mask.as_ref().map(|m| m.as_bytes());
    if single {
        let out = py.detach(|| pyramid::decode::<f32>(b, r, c, h, k, offset, step, q, o, m))?;
        Ok(out.into_pyarray(py).into_any())
    } else {
        let out = py.detach(|| pyramid::decode::<f64>(b, r, c, h, k, offset, step, q, o, m))?;
        Ok(out.into_pyarray(py).into_any())
    }
}

/// The encoder's state between its native parts and the least-squares fit.
#[pyclass(frozen)]
struct PyramidEncoder {
    inner: pyramid::Encoder,
}

#[pymethods]
impl PyramidEncoder {
    #[new]
    fn new(py: Python<'_>, cells: PyReadonlyArray1<'_, f64>, k: usize, q: f64, o: f64) -> PyResult<Self> {
        let c = cells.as_slice()?;
        let inner = py.detach(|| pyramid::Encoder::prepare(c, k, q, o))?;
        Ok(PyramidEncoder { inner })
    }

    /// The parents on the DN lattice, `h x h` uint16.
    fn base_dn<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let h = self.inner.h;
        Ok(PyArray1::from_slice(py, &self.inner.base_dn).reshape([h, h])?.into_any())
    }

    /// `(design, targets)` of the fit on the fully valid parents, or None.
    fn fit_system<'py>(&self, py: Python<'py>) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let kt = self.inner.k * self.inner.k + 1;
        match py.detach(|| self.inner.fit_system()) {
            None => Ok(None),
            Some((design, targets, rows)) => Ok(Some((
                design.into_pyarray(py).reshape([rows, kt])?.into_any(),
                targets.into_pyarray(py).reshape([rows, 4])?.into_any(),
            ))),
        }
    }

    /// `(r16, step, offset, mask_bits)` for the filters `coef` (4 x (k k + 1) float32).
    fn finish<'py>(
        &self,
        py: Python<'py>,
        coef: PyReadonlyArray2<'py, f32>,
        relative_step: f64,
        min_step: f64,
    ) -> PyResult<(Bound<'py, PyAny>, f32, i64, Option<Bound<'py, PyBytes>>)> {
        let c = coef.as_slice()?;
        let (r16, step, offset, mask) = py.detach(|| self.inner.finish(c, relative_step, min_step))?;
        let side = 2 * self.inner.h;
        Ok((
            r16.into_pyarray(py).reshape([side, side])?.into_any(),
            step,
            offset,
            mask.map(|m| PyBytes::new(py, &m)),
        ))
    }
}

#[pymodule]
fn hcc_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("PYRAMID_FORMAT", PYRAMID_FORMAT)?;
    m.add_function(wrap_pyfunction!(pyramid_decode, m)?)?;
    m.add_class::<PyramidEncoder>()?;
    Ok(())
}

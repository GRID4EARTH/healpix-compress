//! Kernels of the two-layer Sentinel-2 coding (`hcc.compression.sentinel2_pyramid`).
//!
//! The Python module is the reference: every floating-point result here is meant
//! to be bit-identical to it, so each operation follows the order numpy and
//! OpenBLAS use there (see `numpy_sum` and `predict_row`).

/// The four children of a parent, as (row, column) offsets in the child raster.
pub const PHASES: [(usize, usize); 4] = [(0, 0), (0, 1), (1, 0), (1, 1)];

/// Digital number reserved for no-data.
pub const NODATA_DN: u16 = 0;

#[derive(Debug)]
pub struct Error(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, Error> {
    Err(Error(msg.into()))
}

// ---------------------------------------------------------------------------
// z-order (Morton) indexing of a 2^d x 2^d raster, as hcc.compression.zorder

/// `even[v]` has bit `b` of `v` at bit `2 b`; cell `(row, col)` of the raster is
/// element `even[row] | even[col] << 1` of the nested chunk.
pub struct ZOrder {
    even: Vec<usize>,
}

impl ZOrder {
    pub fn new(side: usize) -> Self {
        let even = (0..side)
            .map(|v| {
                let mut out = 0usize;
                for b in 0..usize::BITS as usize / 2 {
                    out |= ((v >> b) & 1) << (2 * b);
                }
                out
            })
            .collect();
        ZOrder { even }
    }

    #[inline(always)]
    pub fn index(&self, row: usize, col: usize) -> usize {
        self.even[row] | (self.even[col] << 1)
    }
}

// ---------------------------------------------------------------------------
// numpy's float64 sum

/// `pairwise_sum` of numpy's `loops_utils.h.src`, for float64.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = -0.0;
        for &v in a {
            res += v;
        }
        res
    } else if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let end = n - n % 8;
        let mut i = 8;
        while i < end {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
    }
}

/// `np.add.reduce` of a contiguous 1-D float64 array: the identity plus the
/// pairwise sum of the whole array (numpy 2.x hands it to the inner loop at once).
pub fn numpy_sum(a: &[f64]) -> f64 {
    0.0 + pairwise_sum(a)
}

/// `np.mean` of a contiguous 1-D float64 array.
pub fn numpy_mean(a: &[f64]) -> f64 {
    numpy_sum(a) / a.len() as f64
}

/// `np.std` of a contiguous 1-D float64 array (two passes, as `numpy._core._methods._var`).
pub fn numpy_std(a: &[f64]) -> f64 {
    let n = a.len() as f64;
    let mean = numpy_sum(a) / n;
    let sq: Vec<f64> = a
        .iter()
        .map(|&v| {
            let x = v - mean;
            x * x
        })
        .collect();
    (numpy_sum(&sq) / n).sqrt()
}

// ---------------------------------------------------------------------------
// prediction of the children from the parents

#[inline(always)]
fn fma_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("fma") && is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `np.pad(base, r, mode="edge")` of the `h x h` base, and its side.
fn edge_pad(base: &[f64], h: usize, r: usize) -> (Vec<f64>, usize) {
    let pw = h + 2 * r;
    let mut pad = vec![0.0; pw * pw];
    for i in 0..pw {
        let si = i.saturating_sub(r).min(h - 1);
        for j in 0..pw {
            let sj = j.saturating_sub(r).min(h - 1);
            pad[i * pw + j] = base[si * h + sj];
        }
    }
    (pad, pw)
}

/// The parents' neighbourhoods and the filters of the four phases.
pub struct Predictor {
    k: usize,
    h: usize,
    pw: usize,
    pad: Vec<f64>,
    x: [Vec<f64>; 4],
    acc: Vec<f64>,
}

impl Predictor {
    /// `base` is the `h x h` decoded base, `coef` the `4 x (k k + 1)` float32 filters.
    pub fn new(base: &[f64], h: usize, k: usize, coef: &[f32]) -> Self {
        let (pad, pw) = edge_pad(base, h, k / 2);
        let kt = k * k + 1;
        let x = std::array::from_fn(|p| coef[p * kt..(p + 1) * kt].iter().map(|&v| v as f64).collect());
        Predictor {
            k,
            h,
            pw,
            pad,
            x,
            acc: vec![0.0; 4 * h],
        }
    }

    /// Column `f` of the design matrix (the neighbourhood, then the constant), for parent row `r`.
    #[inline(always)]
    fn column(&self, r: usize, f: usize) -> Option<&[f64]> {
        if f < self.k * self.k {
            let (i, j) = (f / self.k, f % self.k);
            let start = (r + i) * self.pw + j;
            Some(&self.pad[start..start + self.h])
        } else {
            None
        }
    }

    /// The prediction of phase `p` for the parents of row `r`: `design @ coef[p]`
    /// in the order of OpenBLAS `dgemv_t` (Haswell/Zen kernel), which numpy runs
    /// for `_predict`: four FMA lanes over the first `(k k + 1) & !3` columns,
    /// reduced as `(l0 + l2) + (l1 + l3)`, then the two remaining columns as
    /// `y += fma(a0, x0, a1 * x1)` (the C tail, contracted by the compiler).
    /// Checked bit for bit against numpy 2.5 with OpenBLAS 0.3.29 on AMD EPYC
    /// (Zen 3), with one and with 16 OpenBLAS threads.
    pub fn predict_row(&mut self, r: usize, p: usize, out: &mut [f64]) {
        #[cfg(target_arch = "x86_64")]
        {
            if fma_available() {
                // SAFETY: the CPU has AVX2 and FMA
                unsafe { self.predict_row_fma(r, p, out) };
                return;
            }
        }
        self.predict_row_impl(r, p, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn predict_row_fma(&mut self, r: usize, p: usize, out: &mut [f64]) {
        self.predict_row_impl(r, p, out)
    }

    #[inline(always)]
    fn predict_row_impl(&mut self, r: usize, p: usize, out: &mut [f64]) {
        let h = self.h;
        let kt = self.k * self.k + 1;
        let nb = kt & !3;
        let mut acc = std::mem::take(&mut self.acc);
        acc.fill(0.0);
        for f in 0..nb {
            let xf = self.x[p][f];
            let lane = &mut acc[(f & 3) * h..(f & 3) * h + h];
            match self.column(r, f) {
                Some(col) => {
                    for (a, &v) in lane.iter_mut().zip(col) {
                        *a = v.mul_add(xf, *a);
                    }
                }
                None => {
                    for a in lane.iter_mut() {
                        *a = 1.0f64.mul_add(xf, *a);
                    }
                }
            }
        }
        let (l01, l23) = acc.split_at(2 * h);
        let (l0, l1) = l01.split_at(h);
        let (l2, l3) = l23.split_at(h);
        for c in 0..h {
            // y = 0 (beta = 0), then y += ybuffer * alpha
            out[c] = 0.0 + ((l0[c] + l2[c]) + (l1[c] + l3[c]));
        }
        self.acc = acc;
        let m3 = kt & 3;
        match m3 {
            0 => {}
            2 => {
                let (x0, x1) = (self.x[p][nb], self.x[p][nb + 1]);
                let c0 = self.column(r, nb).map(|s| s.to_vec());
                let c1 = self.column(r, nb + 1).map(|s| s.to_vec());
                for c in 0..h {
                    let a0 = c0.as_ref().map_or(1.0, |s| s[c]);
                    let a1 = c1.as_ref().map_or(1.0, |s| s[c]);
                    out[c] += a0.mul_add(x0, a1 * x1);
                }
            }
            _ => {
                // not reached for an odd predictor size (k k + 1 = 2 mod 4)
                for f in nb..kt {
                    let xf = self.x[p][f];
                    let col = self.column(r, f).map(|s| s.to_vec());
                    for c in 0..h {
                        let a = col.as_ref().map_or(1.0, |s| s[c]);
                        out[c] += a * xf;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// base

/// `_decoded_base`: the base as the decoder sees it, no-data parents filled by the mean.
pub fn decoded_base(base_dn: &[u16], q: f64, o: f64) -> Vec<f64> {
    let mut base: Vec<f64> = base_dn
        .iter()
        .map(|&dn| if dn == NODATA_DN { f64::NAN } else { (dn as f64 + o) / q })
        .collect();
    if base.iter().any(|v| !v.is_finite()) {
        let finite: Vec<f64> = base.iter().copied().filter(|v| v.is_finite()).collect();
        let fill = if finite.is_empty() { 0.0 } else { numpy_mean(&finite) };
        for v in base.iter_mut() {
            if !v.is_finite() {
                *v = fill;
            }
        }
    }
    base
}

fn check_side(h: usize) -> Result<(), Error> {
    if h == 0 || !h.is_power_of_two() {
        return err(format!("the base side must be a power of two, got {h}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// decode

pub trait OutFloat: Copy + Send {
    fn from_f64(v: f64) -> Self;
}
impl OutFloat for f64 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }
}
impl OutFloat for f32 {
    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

/// The nested chunk of `4 h^2` cells from the decoded parts of a codestream.
#[allow(clippy::too_many_arguments)]
pub fn decode<T: OutFloat>(
    base_dn: &[u16],
    r16: &[u16],
    coef: &[f32],
    h: usize,
    k: usize,
    offset: i64,
    step: f64,
    q: f64,
    o: f64,
    mask: Option<&[u8]>,
) -> Result<Vec<T>, Error> {
    check_side(h)?;
    let side = 2 * h;
    let kt = k * k + 1;
    if base_dn.len() != h * h || r16.len() != side * side || coef.len() != 4 * kt {
        return err("inconsistent sizes of base, detail and coefficients");
    }
    if let Some(m) = mask {
        if m.len() * 8 < side * side {
            return err(format!("mask of {} bytes for {} cells", m.len(), side * side));
        }
    }
    let base = decoded_base(base_dn, q, o);
    let mut pred = Predictor::new(&base, h, k, coef);
    let z = ZOrder::new(side);
    let mut out = vec![T::from_f64(0.0); side * side];
    let mut row_pred = vec![0.0; h];
    for r in 0..h {
        for (p, &(di, dj)) in PHASES.iter().enumerate() {
            pred.predict_row(r, p, &mut row_pred);
            let row = 2 * r + di;
            for c in 0..h {
                let col = 2 * c + dj;
                let lin = row * side + col;
                let mut valid = base_dn[r * h + c] != NODATA_DN;
                if let Some(m) = mask {
                    valid &= (m[lin >> 3] >> (7 - (lin & 7))) & 1 == 1;
                }
                // pred + (r16 + offset) * step / q, as numpy evaluates it
                let v = if valid {
                    row_pred[c] + ((r16[lin] as i64 + offset) as f64 * step) / q
                } else {
                    f64::NAN
                };
                out[z.index(row, col)] = T::from_f64(v);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// encode

/// The state of an encoding between the parts run natively and the fit.
pub struct Encoder {
    pub h: usize,
    pub k: usize,
    q: f64,
    raster: Vec<f64>,
    count: Vec<u8>,
    pub base_dn: Vec<u16>,
    base: Vec<f64>,
}

impl Encoder {
    /// From the cells: the child raster, the parents' means on the DN lattice and the decoded base.
    pub fn prepare(cells: &[f64], k: usize, q: f64, o: f64) -> Result<Self, Error> {
        let n = cells.len();
        let side = (n as f64).sqrt() as usize;
        if side * side != n || side < 2 {
            return err(format!("a chunk of {n} cells is not a nested chunk of 4**delta, delta >= 1"));
        }
        let h = side / 2;
        check_side(h)?;
        let z = ZOrder::new(side);
        let mut raster = vec![0.0; n];
        for row in 0..side {
            for col in 0..side {
                raster[row * side + col] = cells[z.index(row, col)];
            }
        }
        let mut count = vec![0u8; h * h];
        let mut base_dn = vec![NODATA_DN; h * h];
        for a in 0..h {
            for c in 0..h {
                let at = |b: usize, d: usize| raster[(2 * a + b) * side + 2 * c + d];
                let (x00, x01, x10, x11) = (at(0, 0), at(0, 1), at(1, 0), at(1, 1));
                let cnt = [x00, x01, x10, x11].iter().filter(|v| v.is_finite()).count();
                count[a * h + c] = cnt as u8;
                if cnt > 0 {
                    let f = |v: f64| if v.is_finite() { v } else { 0.0 };
                    // reshape(h, 2, h, 2).sum(axis=(1, 3)): numpy adds along the last
                    // axis first, then the two partial sums
                    let sum = 0.0 + ((f(x00) + f(x01)) + (f(x10) + f(x11)));
                    let mean = sum / cnt as f64;
                    // sentinel2.quantize
                    let scaled = (mean * q - o).round_ties_even();
                    base_dn[a * h + c] = scaled.clamp(1.0, u16::MAX as f64) as u16;
                }
            }
        }
        let base = decoded_base(&base_dn, q, o);
        Ok(Encoder { h, k, q, raster, count, base_dn, base })
    }

    /// The least-squares system of `_fit` (design rows and targets of the fully
    /// valid parents), or `None` when there are fewer rows than unknowns.
    pub fn fit_system(&self) -> Option<(Vec<f64>, Vec<f64>, usize)> {
        let (h, k) = (self.h, self.k);
        let kt = k * k + 1;
        let rows = self.count.iter().filter(|&&c| c == 4).count();
        if rows < kt {
            return None;
        }
        let side = 2 * h;
        let (pad, pw) = edge_pad(&self.base, h, k / 2);
        let mut design = Vec::with_capacity(rows * kt);
        let mut targets = Vec::with_capacity(rows * 4);
        for a in 0..h {
            for c in 0..h {
                if self.count[a * h + c] != 4 {
                    continue;
                }
                for i in 0..k {
                    design.extend_from_slice(&pad[(a + i) * pw + c..(a + i) * pw + c + k]);
                }
                design.push(1.0);
                for &(di, dj) in PHASES.iter() {
                    targets.push(self.raster[(2 * a + di) * side + 2 * c + dj]);
                }
            }
        }
        Some((design, targets, rows))
    }

    /// From the filters: the quantised detail, its step, its offset and the
    /// packed validity bits when some parents are partially valid.
    pub fn finish(
        &self,
        coef: &[f32],
        relative_step: f64,
        min_step: f64,
    ) -> Result<(Vec<u16>, f32, i64, Option<Vec<u8>>), Error> {
        let (h, k) = (self.h, self.k);
        let side = 2 * h;
        if coef.len() != 4 * (k * k + 1) {
            return err("inconsistent size of the coefficients");
        }
        let mut predictor = Predictor::new(&self.base, h, k, coef);
        let mut detail = vec![0.0; side * side];
        let mut row_pred = vec![0.0; h];
        for r in 0..h {
            for (p, &(di, dj)) in PHASES.iter().enumerate() {
                predictor.predict_row(r, p, &mut row_pred);
                let row = 2 * r + di;
                for c in 0..h {
                    let lin = row * side + 2 * c + dj;
                    detail[lin] = (self.raster[lin] - row_pred[c]) * self.q;
                }
            }
        }
        let valid: Vec<bool> = self.raster.iter().map(|v| v.is_finite()).collect();
        let dv: Vec<f64> = detail.iter().zip(&valid).filter(|(_, &ok)| ok).map(|(&d, _)| d).collect();
        let sigma = if dv.is_empty() { 0.0 } else { numpy_std(&dv) };
        let s = relative_step * sigma;
        let step = (if s > min_step { s } else { min_step }) as f32;
        let stepf = step as f64;
        let r: Vec<i64> = detail
            .iter()
            .zip(&valid)
            .map(|(&d, &ok)| if ok { (d / stepf).round_ties_even() as i64 } else { 0 })
            .collect();
        let offset = r.iter().copied().min().unwrap_or(0);
        let r16 = r.iter().map(|&v| (v - offset).clamp(0, u16::MAX as i64) as u16).collect();
        let partial = self.count.iter().any(|&c| c > 0 && c < 4);
        let mask = partial.then(|| {
            let mut bits = vec![0u8; valid.len().div_ceil(8)];
            for (i, &ok) in valid.iter().enumerate() {
                if ok {
                    bits[i >> 3] |= 0x80 >> (i & 7);
                }
            }
            bits
        });
        Ok((r16, step, offset, mask))
    }
}

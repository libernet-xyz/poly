use crate::utils;
use anyhow::{Context, Result, anyhow};
use starkom_bluesky::ThreeAdicField;
use starkom_ff::PrimeField;
use std::any::{Any, TypeId};
use std::collections::BTreeMap;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::{Mutex, OnceLock};

/// Builds the Lagrange basis polynomials returned by [`Polynomial::lagrange0`].
///
/// Running time: O(N).
fn make_lagrange0<F: PrimeField>(n: usize) -> Polynomial<F> {
    let mut coefficients = vec![F::ZERO; n + 1];
    coefficients[0] = -F::ONE;
    coefficients[n] = F::ONE;
    let zero = Polynomial { coefficients };
    let (quotient, remainder) = zero.horner(F::ONE);
    assert_eq!(remainder, F::ZERO);
    quotient * F::try_from(n).unwrap().invert().into_option().unwrap()
}

/// A polynomial expressed as an array of scalar coefficients in ascending degree order (i.e. the
/// first coefficient is the constant term).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Polynomial<F: PrimeField> {
    coefficients: Vec<F>,
}

impl<F: PrimeField> Polynomial<F> {
    /// Constructs a polynomial with the provided coefficients, which must be in ascending degree
    /// order.
    pub fn with_coefficients(coefficients: Vec<F>) -> Self {
        Self { coefficients }
    }

    /// Returns a zero-degree polynomial that evaluates to `y` everywhere.
    pub fn constant(y: F) -> Self {
        Self {
            coefficients: vec![y],
        }
    }

    /// Constructs a polynomial that interpolates the given points using Lagrange interpolation.
    ///
    /// The points are specified as (x, y) pairs.
    ///
    /// Running time: O(N^2).
    pub fn interpolate(points: &[(F, F)]) -> Result<Self> {
        let k = points.len();
        let x = points.iter().map(|(x, _)| *x).collect::<Vec<F>>();
        let l = Self::from_roots(x.as_slice(), F::ONE).context("duplicate X-coordinates")?;
        let w = {
            let one = F::ONE;
            let mut weights = vec![one; k];
            for i in 0..k {
                for j in 0..k {
                    if i != j {
                        weights[i] *= x[i] - x[j];
                    }
                }
                weights[i] = weights[i]
                    .invert()
                    .into_option()
                    .context("duplicate X-coordinates")?;
            }
            weights
        };
        let mut result = Self {
            coefficients: Vec::with_capacity(points.len()),
        };
        for i in 0..k {
            let (basis, remainder) = l.horner(x[i]);
            assert_eq!(remainder, F::ZERO);
            let (_, y) = points[i];
            result += basis * w[i] * y;
        }
        Ok(result)
    }

    /// Interpolates a polynomial that has the given roots.
    ///
    /// This algorithm is roughly twice faster than simply calling [`Self::interpolate`] with 0 as
    /// the y coordinate of all points.
    ///
    /// NOTE: if the caller's protocol doesn't require a blinding factor it can be set to 1. Do NOT
    /// set it to 0, as that would nullify the whole polynomial.
    ///
    /// Running time: O(N^2).
    pub fn from_roots(roots: &[F], blinding_factor: F) -> Result<Self> {
        let mut roots = roots.to_vec();
        roots.sort();
        for i in 1..roots.len() {
            if roots[i] == roots[i - 1] {
                return Err(anyhow!("duplicate roots"));
            }
        }
        let n = roots.len() + 1;
        let mut coefficients = vec![F::ZERO; n];
        coefficients[0] = blinding_factor;
        for i in 1..n {
            for j in (0..i).rev() {
                let c = coefficients[j];
                coefficients[j + 1] -= c * roots[i - 1];
            }
        }
        coefficients.reverse();
        Ok(Self { coefficients })
    }

    /// 2-adic Fast Fourier Transform.
    ///
    /// REQUIRES: the length of `data` must be a power of two less than or equal to N and `omega`
    /// must be an N-th root of unity, where N = 2^(F::S).
    ///
    /// Running time: O(N*logN).
    fn fft2(data: &mut [F], omega: F) {
        let n = data.len();
        assert!(n.is_power_of_two());

        let log_n = n.trailing_zeros();
        assert!(log_n as usize <= F::S);

        for i in 0..n {
            let (j, _) = i.reverse_bits().overflowing_shr(usize::BITS - log_n);
            if i < j {
                data.swap(i, j);
            }
        }

        let mut m = 1;
        for _ in 0..log_n {
            let step = m * 2;
            let wm = omega.pow_small(n / step);
            let mut w = F::ONE;
            for k in 0..m {
                for j in (k..n).step_by(step) {
                    let t = w * data[j + m];
                    let u = data[j];
                    data[j] = u + t;
                    data[j + m] = u - t;
                }
                w *= wm;
            }
            m = step;
        }
    }

    /// Inverse 2-adic Fast Fourier Transform.
    ///
    /// REQUIRES: `n` must be a power of two less than or equal to 2^S, with `S` being the 2-adicity
    /// of the field `F` (supplied as `F::S`).
    ///
    /// Running time: O(N*logN).
    fn ifft2(data: &mut [F], omega: F) {
        Self::fft2(data, omega.invert().into_option().unwrap());
        let n_inv = F::try_from(data.len()).unwrap().invert().unwrap();
        for v in data.iter_mut() {
            *v *= n_inv;
        }
    }

    /// Computes an N-th root of unity where N is a power of 2 less than or equal to 2^(F::S).
    fn two_adic_root_of_unity(n: usize) -> F {
        assert!(n.is_power_of_two());
        let k = n.trailing_zeros() as usize;
        assert!(k <= F::S);
        let exponent = 1u64 << (F::S - k);
        F::ROOT_OF_UNITY.pow_u64(exponent)
    }

    /// Interpolates a polynomial that encodes an ordered list of values.
    ///
    /// The returned polynomial evaluates to the provided values at certain powers of
    /// `F::ROOT_OF_UNITY`. The exact coordinates can be retrieved by calling
    /// [`Self::domain_element2`] with the index of the value to query and the size of the domain
    /// (i.e. `values.len()`).
    ///
    /// NOTE: this function is called `encode2` because it uses the two-adic evaluation domain. For
    /// the three-adic version see [`Self::encode3`] below.
    ///
    /// Under the hood we use the two-adic Inverse Fourier Transform algorithm ([`Self::ifft2`]),
    /// which requires the size of the list to be a power of two. If that's not the case, this
    /// function will automatically pad the provided list with zeros.
    ///
    /// Additionally, the provided list must not exceed the FFT capacity so it's required to have no
    /// more than 2^(F::S) elements.
    ///
    /// Running time: O(N*logN).
    pub fn encode2(mut values: Vec<F>) -> Self {
        assert!(!values.is_empty());
        let n = values.len().next_power_of_two();
        assert!(n.trailing_zeros() as usize <= F::S);
        values.resize(n, F::ZERO);
        let omega = Self::two_adic_root_of_unity(values.len());
        Self::ifft2(values.as_mut_slice(), omega);
        let mut polynomial = Polynomial {
            coefficients: values,
        };
        polynomial.trim();
        polynomial
    }

    /// Recovers the ordered list of values encoded by [`Self::encode2`].
    ///
    /// This is the inverse of [`Self::encode2`]: given a polynomial produced by `encode2(values)`,
    /// calling `decode2` returns a list equal to `values` (possibly padded with trailing zeros to
    /// the next power of two).
    ///
    /// Under the hood we use the two-adic Fast Fourier Transform algorithm ([`Self::fft2`]). The
    /// polynomial's coefficient list is zero-padded to the next power of two before the transform
    /// is applied.
    ///
    /// Running time: O(N*logN).
    pub fn decode2(self) -> Vec<F> {
        let mut data = self.coefficients;
        let n = data.len().next_power_of_two();
        data.resize(n, F::ZERO);
        let omega = Self::two_adic_root_of_unity(n);
        Self::fft2(&mut data, omega);
        data
    }

    /// Returns the number of coefficients, which is equal to the maximum degree plus 1.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns the coefficients of the polynomial in ascending degree order.
    pub fn coefficients(&self) -> &[F] {
        self.coefficients.as_slice()
    }

    fn degree_bound_of(coefficients: &[F]) -> usize {
        for (i, &coefficient) in coefficients.iter().enumerate().rev() {
            if coefficient != F::ZERO {
                return i + 1;
            }
        }
        0
    }

    /// Returns the degree bound of the polynomial, ie. the smallest number `d` such that the degree
    /// is strcitly less than `d`.
    ///
    /// Equivalently: this function returns the degree plus one.
    ///
    /// Running time: O(N) due to the possibility that some of the trailing coefficients are zero.
    pub fn degree_bound(&self) -> usize {
        Self::degree_bound_of(self.coefficients.as_slice())
    }

    /// Removes any trailing null coefficients.
    ///
    /// After this call, [`Self::len()`] is guaranteed to reflect the actual degree bound of the
    /// polynomial:
    ///
    ///   poly.trim();
    ///   assert_eq!(poly.len(), poly.degree_bound());
    pub fn trim(&mut self) {
        if let Some(i) = self
            .coefficients
            .iter()
            .rposition(|value| *value != F::ZERO)
        {
            self.coefficients.truncate(i + 1);
        } else {
            self.coefficients.clear();
        }
    }

    /// Pads the polynomial with null coefficients until the degree bound is at least
    /// `degree_bound`.
    pub fn pad(&mut self, min_degree_bound: usize) {
        let new_length = std::cmp::max(min_degree_bound, self.coefficients.len());
        self.coefficients.resize(new_length, F::ZERO);
    }

    /// Extracts the array of coefficients from this polynomial.
    ///
    /// NOTE: the coefficients are in ascending degree order, i.e. the first returned element is the
    /// constant term.
    pub fn take(self) -> Vec<F> {
        return self.coefficients;
    }

    /// Multiplies two polynomials. Panics if the FFT capacity is exceeded -- that is, if the degree
    /// of the product is greater than or equal to 2^(F::S).
    pub fn multiply(mut self, mut other: Self) -> Self {
        self.trim();
        other.trim();

        let mut lhs = self.coefficients;
        let mut rhs = other.coefficients;

        if lhs.is_empty() || rhs.is_empty() {
            return Polynomial {
                coefficients: vec![],
            };
        }
        if lhs.len() == 1 {
            return Polynomial { coefficients: rhs } * lhs[0];
        }
        if rhs.len() == 1 {
            return Polynomial { coefficients: lhs } * rhs[0];
        }

        let n = (lhs.len() + rhs.len() - 1).next_power_of_two();

        lhs.resize(n, F::ZERO);
        rhs.resize(n, F::ZERO);

        let omega = Self::two_adic_root_of_unity(n);
        Self::fft2(lhs.as_mut_slice(), omega);
        Self::fft2(rhs.as_mut_slice(), omega);

        for i in 0..n {
            lhs[i] *= rhs[i];
        }

        Self::ifft2(lhs.as_mut_slice(), omega);

        let mut result = Polynomial { coefficients: lhs };
        result.trim();
        result
    }

    /// Internal implementation of [`Self::multiply_many`].
    fn multiply_many_impl(polynomials: &mut [Self]) -> Self {
        match polynomials.len() {
            0 => Polynomial {
                coefficients: vec![],
            },
            1 => std::mem::take(&mut polynomials[0]),
            2 => {
                let lhs = std::mem::take(&mut polynomials[0]);
                let rhs = std::mem::take(&mut polynomials[1]);
                lhs.multiply(rhs)
            }
            n => {
                let (left, right) = polynomials.split_at_mut(n / 2);
                let left = Self::multiply_many_impl(left);
                let right = Self::multiply_many_impl(right);
                left.multiply(right)
            }
        }
    }

    /// Multiplies two or more polynomials, returning an error if the FFT capacity is exceeded --
    /// that is, if the degree of the product is greater than or equal to 2^(F::S).
    ///
    /// REQUIRES: the `polynomials` array must have at least 1 element, otherwise the function will
    /// panic.
    pub fn multiply_many<const N: usize>(mut polynomials: [Self; N]) -> Self {
        assert!(N > 0);
        Self::multiply_many_impl(&mut polynomials)
    }

    /// Multiplies two polynomials defined on the value domain, assuming the provided evaluations
    /// are defined on the same two-adic evaluation domain for both.
    ///
    /// REQUIRES: the LHS and RHS must have the same length `n` and it must be a power of two. The
    /// implied evaluation domain is the set of powers of an `n`-th root of unity.
    ///
    /// The returned polynomial is also on the value domain and can be switched to the coefficient
    /// domain by constructing a [`Polynomial`] object on it (see [`Self::encode2`]).
    pub fn multiply_values2(mut lhs: Vec<F>, mut rhs: Vec<F>) -> Vec<F> {
        let n = lhs.len();
        assert!(n.is_power_of_two());
        assert!(n.trailing_zeros() as usize + 1 <= F::S);
        assert_eq!(rhs.len(), n);
        let omega = Self::two_adic_root_of_unity(n);
        Self::ifft2(&mut lhs, omega);
        Self::ifft2(&mut rhs, omega);
        let lhs_len = Self::degree_bound_of(lhs.as_slice());
        let rhs_len = Self::degree_bound_of(rhs.as_slice());
        let m = (lhs_len + rhs_len - 1).next_power_of_two();
        lhs.resize(m, F::ZERO);
        rhs.resize(m, F::ZERO);
        let omega = Self::two_adic_root_of_unity(m);
        Self::fft2(&mut lhs, omega);
        Self::fft2(&mut rhs, omega);
        for i in 0..m {
            lhs[i] *= rhs[i];
        }
        lhs
    }

    /// Divides this polynomial by (x - z) using Horner's method. Returns the quotient polynomial
    /// and the remainder scalar.
    ///
    /// Running time: O(N).
    pub fn horner(&self, z: F) -> (Self, F) {
        if self.coefficients.is_empty() {
            return (Polynomial::default(), F::ZERO);
        }
        let n = self.len() - 1;
        let mut coefficients = vec![F::ZERO; n];
        if n < 1 {
            return (Polynomial { coefficients }, self.coefficients[0]);
        }
        coefficients[n - 1] = self.coefficients[n];
        for i in (1..n).rev() {
            coefficients[i - 1] = self.coefficients[i] + z * coefficients[i];
        }
        let remainder = self.coefficients[0] + z * coefficients[0];
        (Polynomial { coefficients }, remainder)
    }

    /// Divides this polynomial by (x^n - 1), succeeding only if the remainder is 0. The polynomial
    /// wrapped in a successful result is the quotient Q such that Q(x) * (x^n - 1) equals this
    /// polynomial.
    ///
    /// Note that (x^n - 1) is a polynomial that evaluates to zero across an evaluation domain of
    /// size `n`, because the roots of it are the n-th roots of unity. We call this the "zero
    /// polynomial".
    ///
    /// NOTE: this algorithm doesn't check that `n` is a power of 2 and will work with arbitrary
    /// values of `n`, but it's generally most useful when `n` is a power of 2.
    ///
    /// Running time: O(N).
    pub fn divide_by_zero(&self, n: usize) -> Result<Self> {
        let mut data = self.coefficients.clone();
        if data.len() < n {
            data.resize(n, F::ZERO);
        }

        let degree = data.len() - n;
        let mut quotient = vec![F::ZERO; degree];

        let neg_one = F::ZERO - F::ONE;
        for i in 0..degree {
            let c = data[i] * neg_one;
            quotient[i] = c;
            data[i] += c;
            data[i + n] -= c;
        }

        let remainder = &data[degree..];
        if remainder.iter().any(|c| *c != F::ZERO) {
            return Err(anyhow!("non-zero remainder in division by (x^n - 1)"));
        }

        if let Some(i) = quotient.iter().rposition(|c| *c != F::ZERO) {
            quotient.truncate(i + 1);
        }
        Ok(Polynomial {
            coefficients: quotient,
        })
    }

    /// Evaluates the polynomial at the specified X coordinate.
    ///
    /// Running time: O(N).
    ///
    /// NOTE: the returned value is the same as the remainder value returned by the [`Self::horner`]
    /// algorithm above. Even though the two algorithms have the same asymptotic running time, this
    /// one is faster because it doesn't allocate memory for the quotient polynomial.
    pub fn evaluate(&self, x: F) -> F {
        let mut y = F::ZERO;
        for coefficient in self.coefficients.iter().rev() {
            y = y * x + *coefficient;
        }
        y
    }

    /// Returns the X coordinate of the i-th element of a list encoded with [`Self::encode2`].
    ///
    /// The returned value is suitable for use with [`Self::evaluate`] to query the original value
    /// from the encoded list.
    ///
    /// `domain_size` is the length of the original list. It will be rounded up to the next power of
    /// two automatically.
    ///
    /// Running time: O(1).
    pub fn domain_element2(index: usize, domain_size: usize) -> F {
        let omega = Self::two_adic_root_of_unity(domain_size.next_power_of_two());
        omega.pow_small(index)
    }

    /// Returns the X coordinate of the i-th point in the coset LDE domain used by
    /// [`Self::shifted_lde2`].
    ///
    /// Equivalent to `F::MULTIPLICATIVE_GENERATOR * domain_element2(index, domain_size)`.
    ///
    /// Running time: O(1).
    pub fn coset_element2(index: usize, domain_size: usize) -> F {
        F::MULTIPLICATIVE_GENERATOR * Self::domain_element2(index, domain_size)
    }

    /// Same as `evaluate(domain_element2(index, domain_size))`.
    ///
    /// Running time: O(N).
    pub fn evaluate_on_two_adic_domain(&self, index: usize, domain_size: usize) -> F {
        self.evaluate(Self::domain_element2(index, domain_size))
    }

    /// Same as `evaluate(coset_element2(index, domain_size))`.
    ///
    /// Running time: O(N).
    pub fn evaluate_on_two_adic_coset(&self, index: usize, domain_size: usize) -> F {
        self.evaluate(Self::coset_element2(index, domain_size))
    }

    /// Computes a low-degree extension of the polynomial by evaluating it at `m` points on the
    /// coset `shift * <omega_m>`, where `omega_m` is a primitive `m`-th root of unity and `shift`
    /// is the multiplicative generator of the field, `F::MULTIPLICATIVE_GENERATOR`. The evaluation
    /// points are `shift * omega_m^i` for `i = 0..m`.
    ///
    /// The algorithm shifts the evaluation domain so that the resulting values can be used in
    /// (DEEP-)FRI without revealing any of the original values. The coset shift is applied by
    /// multiplying each coefficient `a_k` by `F::MULTIPLICATIVE_GENERATOR^k` before the FFT, which
    /// is equivalent to substituting `X -> shift * X` in the polynomial.
    ///
    /// REQUIRES: `m` must be a power of two at least as large as `self.len()`, and no larger than
    /// `2^(F::S)`.
    ///
    /// Running time: O(M*log(M)).
    pub fn shifted_lde2(self, m: usize) -> Vec<F> {
        assert!(m.is_power_of_two());
        assert!(m.trailing_zeros() as usize <= F::S);
        assert!(self.coefficients.len() <= m);
        let mut data = self.coefficients;
        data.resize(m, F::ZERO);
        let mut shift_pow = F::ONE;
        for c in data.iter_mut() {
            *c *= shift_pow;
            shift_pow *= F::MULTIPLICATIVE_GENERATOR;
        }
        let omega = Self::two_adic_root_of_unity(m);
        Self::fft2(&mut data, omega);
        data
    }

    /// Folding algorithm used in FRI and similar algorithms.
    ///
    /// `alpha` is a verifier challenge, typically derived via Fiat-Shamir.
    pub fn fold2(self, alpha: F) -> Self {
        let coefficients = self.coefficients();
        let m = (coefficients.len() + 1) / 2;
        let new_coefficients = (0..m)
            .map(|i| {
                coefficients[2 * i]
                    + alpha * coefficients.get(2 * i + 1).copied().unwrap_or(F::ZERO)
            })
            .collect();
        Self::with_coefficients(new_coefficients)
    }
}

impl<F: PrimeField + ThreeAdicField> Polynomial<F> {
    /// 3-adic Fast Fourier Transform.
    ///
    /// REQUIRES: the length of `data` must be a power of three less than or equal to N and `omega`
    /// must be an N-th root of unity, where N = 3^(F::T).
    ///
    /// Running time: O(N*logN).
    fn fft3(data: &mut [F], omega: F) {
        let n = data.len();
        assert!(utils::is_power_of_three(n));

        let log_n = utils::ilog3(n);

        for i in 0..n {
            let mut j = 0;
            let mut tmp = i;
            for _ in 0..log_n {
                j = j * 3 + tmp % 3;
                tmp /= 3;
            }
            if i < j {
                data.swap(i, j);
            }
        }

        let omega3 = omega.pow_small(n / 3);
        let omega3_sq = omega3 * omega3;

        let mut m = 1;
        for _ in 0..log_n {
            let step = m * 3;
            let wm = omega.pow_small(n / step);
            let mut w = F::ONE;
            let mut w2 = F::ONE;
            for k in 0..m {
                for j in (k..n).step_by(step) {
                    let t0 = data[j];
                    let t1 = w * data[j + m];
                    let t2 = w2 * data[j + 2 * m];
                    data[j] = t0 + t1 + t2;
                    data[j + m] = t0 + omega3 * t1 + omega3_sq * t2;
                    data[j + 2 * m] = t0 + omega3_sq * t1 + omega3 * t2;
                }
                w *= wm;
                w2 = w * w;
            }
            m = step;
        }
    }

    /// Inverse 3-adic Fast Fourier Transform.
    ///
    /// REQUIRES: the length of `data` must be a power of three less than or equal to 3^(F::T), with
    /// `T` being the 3-adicity of the field `F` (supplied as `F::T`).
    ///
    /// Running time: O(N*logN).
    fn ifft3(data: &mut [F], omega: F) {
        Self::fft3(data, omega.invert().into_option().unwrap());
        let n_inv = F::try_from(data.len()).unwrap().invert().unwrap();
        for v in data.iter_mut() {
            *v *= n_inv;
        }
    }

    /// Computes an N-th root of unity where N is a power of 3 less than or equal to 3^(F::T).
    fn three_adic_root_of_unity(n: usize) -> F {
        assert!(utils::is_power_of_three(n));
        let k = utils::ilog3(n) as u32;
        assert!(k <= F::T);
        let exponent = 3u64.pow(F::T - k);
        F::THREE_ADIC_ROOT_OF_UNITY.pow_u64(exponent)
    }

    /// Interpolates a polynomial that encodes an ordered list of values.
    ///
    /// The returned polynomial evaluates to the provided values at certain powers of the
    /// `F::THREE_ADIC_ROOT_OF_UNITY`. The exact coordinates can be retrieved by calling
    /// [`Self::domain_element3`] with the index of the value to query and the size of the domain
    /// (i.e. `values.len()`).
    ///
    /// NOTE: this function is called `encode3` because it uses the three-adic evaluation domain.
    /// For the two-adic version see [`Self::encode2`] above.
    ///
    /// Under the hood we use the three-adic Inverse Fourier Transform algorithm ([`Self::ifft3`]),
    /// which requires the size of the list to be a power of three. If that's not the case, this
    /// function will automatically pad the provided list with zeros.
    ///
    /// Additionally, the provided list must not exceed the FFT capacity so it's required to have no
    /// more than 3^(F::T) elements.
    ///
    /// Running time: O(N*logN).
    pub fn encode3(mut values: Vec<F>) -> Self {
        assert!(!values.is_empty());
        let n = utils::next_power_of_three(values.len());
        assert!(utils::ilog3(n) <= F::T as usize);
        values.resize(n, F::ZERO);
        let omega = Self::three_adic_root_of_unity(values.len());
        Self::ifft3(values.as_mut_slice(), omega);
        let mut polynomial = Polynomial {
            coefficients: values,
        };
        polynomial.trim();
        polynomial
    }

    /// Recovers the ordered list of values encoded by [`Self::encode3`].
    ///
    /// This is the inverse of [`Self::encode3`]: given a polynomial produced by `encode3(values)`,
    /// calling `decode3` returns a list equal to `values` (possibly padded with trailing zeros to
    /// the next power of three).
    ///
    /// Under the hood we use the three-adic Fast Fourier Transform algorithm ([`Self::fft3`]). The
    /// polynomial's coefficient list is zero-padded to the next power of three before the transform
    /// is applied.
    ///
    /// Running time: O(N*logN).
    pub fn decode3(self) -> Vec<F> {
        let mut data = self.coefficients;
        let n = utils::next_power_of_three(data.len());
        data.resize(n, F::ZERO);
        let omega = Self::three_adic_root_of_unity(n);
        Self::fft3(&mut data, omega);
        data
    }

    /// Returns the X coordinate of the i-th element of a list encoded with [`Self::encode3`].
    ///
    /// The returned value is suitable for use with [`Self::evaluate`] to query the original value
    /// from the encoded list.
    ///
    /// `domain_size` is the length of the original list. It will be rounded up to the next power of
    /// three automatically.
    ///
    /// Running time: O(1).
    pub fn domain_element3(index: usize, domain_size: usize) -> F {
        let omega = Self::three_adic_root_of_unity(utils::next_power_of_three(domain_size));
        omega.pow_small(index)
    }

    /// Returns the X coordinate of the i-th point in the coset LDE domain used by
    /// [`Self::shifted_lde3`].
    ///
    /// Equivalent to `F::MULTIPLICATIVE_GENERATOR * domain_element3(index, domain_size)`.
    ///
    /// Running time: O(1).
    pub fn coset_element3(index: usize, domain_size: usize) -> F {
        F::MULTIPLICATIVE_GENERATOR * Self::domain_element3(index, domain_size)
    }

    /// Same as `evaluate(domain_element3(index, domain_size))`.
    ///
    /// Running time: O(N).
    pub fn evaluate_on_three_adic_domain(&self, index: usize, domain_size: usize) -> F {
        self.evaluate(Self::domain_element3(index, domain_size))
    }

    /// Same as `evaluate(coset_element3(index, domain_size))`.
    ///
    /// Running time: O(N).
    pub fn evaluate_on_three_adic_coset(&self, index: usize, domain_size: usize) -> F {
        self.evaluate(Self::coset_element3(index, domain_size))
    }

    /// Computes a low-degree extension of the polynomial by evaluating it at `m` points on the
    /// coset `shift * <omega_m>`, where `omega_m` is a primitive `m`-th root of unity and `shift`
    /// is the multiplicative generator of the field, `F::MULTIPLICATIVE_GENERATOR`. The evaluation
    /// points are `shift * omega_m^i` for `i = 0..m`.
    ///
    /// The algorithm shifts the evaluation domain so that the resulting values can be used in
    /// (DEEP-)FRI without revealing any of the original values. The coset shift is applied by
    /// multiplying each coefficient `a_k` by `F::MULTIPLICATIVE_GENERATOR^k` before the FFT, which
    /// is equivalent to substituting `X -> shift * X` in the polynomial.
    ///
    /// REQUIRES: `m` must be a power of three at least as large as `self.len()`, and no larger than
    /// `3^(F::T)`.
    ///
    /// Running time: O(M*log(M)).
    pub fn shifted_lde3(self, m: usize) -> Vec<F> {
        assert!(utils::is_power_of_three(m));
        assert!(utils::ilog3(m) as u32 <= F::T);
        assert!(self.coefficients.len() <= m);
        let mut data = self.coefficients;
        data.resize(m, F::ZERO);
        let mut shift_pow = F::ONE;
        for c in data.iter_mut() {
            *c *= shift_pow;
            shift_pow *= F::MULTIPLICATIVE_GENERATOR;
        }
        let omega = Self::three_adic_root_of_unity(m);
        Self::fft3(&mut data, omega);
        data
    }

    /// Folding algorithm used in three-adic FRI and similar algorithms.
    ///
    /// `alpha` is a verifier challenge, typically derived via Fiat-Shamir.
    pub fn fold3(self, alpha: F) -> Self {
        let coefficients = self.coefficients();
        let m = (coefficients.len() + 2) / 3;
        let alpha_square = alpha * alpha;
        let new_coefficients = (0..m)
            .map(|i| {
                coefficients[3 * i]
                    + alpha * coefficients.get(3 * i + 1).copied().unwrap_or(F::ZERO)
                    + alpha_square * coefficients.get(3 * i + 2).copied().unwrap_or(F::ZERO)
            })
            .collect();
        Self::with_coefficients(new_coefficients)
    }

    /// Multiplies two polynomials defined on the value domain, assuming the provided evaluations
    /// are defined on the same three-adic evaluation domain for both.
    ///
    /// REQUIRES: the LHS and RHS must have the same length `n` and it must be a power of three.
    /// The implied evaluation domain is the set of powers of an `n`-th root of unity.
    ///
    /// The returned polynomial is also on the value domain and can be switched to the coefficient
    /// domain by constructing a [`Polynomial`] object on it (see [`Self::encode3`]).
    pub fn multiply_values3(mut lhs: Vec<F>, mut rhs: Vec<F>) -> Vec<F> {
        let n = lhs.len();
        assert!(utils::is_power_of_three(n));
        assert!(utils::ilog3(n) as u32 + 1 <= F::T);
        assert_eq!(rhs.len(), n);
        let omega = Self::three_adic_root_of_unity(n);
        Self::ifft3(&mut lhs, omega);
        Self::ifft3(&mut rhs, omega);
        let lhs_len = Self::degree_bound_of(lhs.as_slice());
        let rhs_len = Self::degree_bound_of(rhs.as_slice());
        let m = utils::next_power_of_three(lhs_len + rhs_len - 1);
        lhs.resize(m, F::ZERO);
        rhs.resize(m, F::ZERO);
        let omega = Self::three_adic_root_of_unity(m);
        Self::fft3(&mut lhs, omega);
        Self::fft3(&mut rhs, omega);
        for i in 0..m {
            lhs[i] *= rhs[i];
        }
        lhs
    }

    /// Returns the Lagrange basis polynomial L0 that activates on the first point of the evaluation
    /// domain of size `n` and evaluates to 0 over the rest.
    ///
    /// In other words:
    ///
    ///   L0(1) = 1
    ///   L0(w^i) = 0 for all i != 0, i < n
    ///
    /// where `w` is an n-th root of unity.
    ///
    /// REQUIRES: `n` must be a power of 2 less than or equal to 2^(F::S).
    ///
    /// These polynomials are used in the PLONK proving scheme running over BlueSky. BlueSky
    /// supports at most 62 of these. Computed on first use and cached for the lifetime of the
    /// program.
    pub fn lagrange0(n: usize) -> &'static Self {
        assert!(n.is_power_of_two());
        let k = n.trailing_zeros() as usize;
        assert!(k <= F::S);

        static CACHE: OnceLock<Mutex<BTreeMap<(TypeId, usize), &'static (dyn Any + Send + Sync)>>> =
            OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));

        let polynomial = {
            let mut map = cache.lock().unwrap();
            *map.entry((TypeId::of::<F>(), k)).or_insert_with(|| {
                Box::leak(Box::new(make_lagrange0::<F>(1 << k))) as &'static (dyn Any + Send + Sync)
            })
        };

        polynomial.downcast_ref::<Polynomial<F>>().unwrap()
    }
}

impl<F: PrimeField> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        for coefficient in &mut self.coefficients {
            *coefficient = -*coefficient;
        }
        self
    }
}

impl<F: PrimeField> Add<Polynomial<F>> for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        if rhs.len() > self.len() {
            return rhs + self;
        }
        for i in 0..rhs.len() {
            self.coefficients[i] += rhs.coefficients[i];
        }
        self
    }
}

impl<F: PrimeField> AddAssign<Polynomial<F>> for Polynomial<F> {
    fn add_assign(&mut self, mut rhs: Self) {
        if rhs.len() > self.len() {
            for i in 0..self.len() {
                rhs.coefficients[i] += self.coefficients[i];
            }
            self.coefficients = rhs.coefficients;
        } else {
            for i in 0..rhs.len() {
                self.coefficients[i] += rhs.coefficients[i];
            }
        }
    }
}

impl<F: PrimeField> Add<F> for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: F) -> Self::Output {
        if self.coefficients.is_empty() {
            self.coefficients.push(rhs);
        } else {
            self.coefficients[0] += rhs;
        }
        self
    }
}

impl<F: PrimeField> AddAssign<F> for Polynomial<F> {
    fn add_assign(&mut self, rhs: F) {
        if self.coefficients.is_empty() {
            self.coefficients.push(rhs);
        } else {
            self.coefficients[0] += rhs;
        }
    }
}

impl<F: PrimeField> Sub<Polynomial<F>> for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        if rhs.len() > self.len() {
            return -(rhs - self);
        }
        for i in 0..rhs.len() {
            self.coefficients[i] -= rhs.coefficients[i];
        }
        self
    }
}

impl<F: PrimeField> SubAssign<Polynomial<F>> for Polynomial<F> {
    fn sub_assign(&mut self, mut rhs: Self) {
        if rhs.len() > self.len() {
            for i in 0..self.len() {
                rhs.coefficients[i] -= self.coefficients[i];
            }
            self.coefficients = rhs.coefficients;
            for i in 0..self.len() {
                self.coefficients[i] = -self.coefficients[i];
            }
        } else {
            for i in 0..rhs.len() {
                self.coefficients[i] -= rhs.coefficients[i];
            }
        }
    }
}

impl<F: PrimeField> Sub<F> for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: F) -> Self::Output {
        if self.coefficients.is_empty() {
            self.coefficients.push(-rhs);
        } else {
            self.coefficients[0] -= rhs;
        }
        self
    }
}

impl<F: PrimeField> SubAssign<F> for Polynomial<F> {
    fn sub_assign(&mut self, rhs: F) {
        if self.coefficients.is_empty() {
            self.coefficients.push(-rhs);
        } else {
            self.coefficients[0] -= rhs;
        }
    }
}

impl<F: PrimeField> Mul<F> for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        for i in 0..self.len() {
            self.coefficients[i] *= rhs;
        }
        self
    }
}

impl<F: PrimeField> MulAssign<F> for Polynomial<F> {
    fn mul_assign(&mut self, rhs: F) {
        for i in 0..self.len() {
            self.coefficients[i] *= rhs;
        }
    }
}

impl<F: PrimeField> Mul<Polynomial<F>> for Polynomial<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.multiply(rhs)
    }
}

impl<F: PrimeField> MulAssign<Polynomial<F>> for Polynomial<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = std::mem::take(self).multiply(rhs);
    }
}

#[cfg(test)]
mod tests {
    use starkom_bluesky::Scalar;
    use starkom_ff::Field;

    type Polynomial = super::Polynomial<Scalar>;

    fn get_random_scalar() -> Scalar {
        Scalar::random_default()
    }

    fn from_roots(roots: &[Scalar]) -> Polynomial {
        Polynomial::from_roots(roots, get_random_scalar()).unwrap()
    }

    #[test]
    fn test_constant() {
        let p = Polynomial::constant(Scalar::from_const(42));
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(42));
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(42));
        assert_eq!(p.evaluate(Scalar::from_const(42)), Scalar::from_const(42));
    }

    #[test]
    fn test_zero() {
        let p = Polynomial::with_coefficients(vec![]);
        assert_eq!(p, Polynomial::default());
        assert_eq!(p.len(), 0);
        assert_eq!(p.degree_bound(), 0);
        assert_eq!(p.evaluate(Scalar::from_const(42)), Scalar::from_const(0));
    }

    #[test]
    fn test_with_coefficients() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(p.len(), 3);
        assert_eq!(p.degree_bound(), 3);
        assert_eq!(
            p.take(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56)
            ]
        );
    }

    #[test]
    fn test_low_degree() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(0),
            Scalar::from_const(0),
        ]);
        assert_eq!(p.len(), 5);
        assert_eq!(p.degree_bound(), 3);
    }

    #[test]
    fn test_skip_degree() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(p.len(), 5);
        assert_eq!(p.degree_bound(), 5);
    }

    #[test]
    fn test_trim_degree() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(0),
            Scalar::from_const(0),
        ]);
        p.trim();
        assert_eq!(p.len(), 3);
        assert_eq!(p.degree_bound(), 3);
    }

    #[test]
    fn test_no_trim() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        p.trim();
        assert_eq!(p.len(), 5);
        assert_eq!(p.degree_bound(), 5);
    }

    #[test]
    fn test_trim_all_zero() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(0),
        ]);
        p.trim();
        assert_eq!(p.len(), p.degree_bound());
        assert_eq!(p, Polynomial::default());
    }

    #[test]
    fn test_pad_extends() {
        let mut p =
            Polynomial::with_coefficients(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        p.pad(5);
        assert_eq!(p.len(), 5);
        assert_eq!(
            p.take(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(0),
                Scalar::from_const(0),
                Scalar::from_const(0)
            ]
        );
    }

    #[test]
    fn test_pad_exact() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        p.pad(3);
        assert_eq!(p.len(), 3);
        assert_eq!(
            p.take(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56)
            ]
        );
    }

    #[test]
    fn test_pad_no_shrink() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
        ]);
        p.pad(2);
        assert_eq!(p.len(), 4);
        assert_eq!(
            p.take(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56),
                Scalar::from_const(78)
            ]
        );
    }

    #[test]
    fn test_pad_empty() {
        let mut p = Polynomial::default();
        p.pad(3);
        assert_eq!(p.len(), 3);
        assert_eq!(
            p.take(),
            vec![
                Scalar::from_const(0),
                Scalar::from_const(0),
                Scalar::from_const(0)
            ]
        );
    }

    #[test]
    fn test_pad_zero_bound() {
        let mut p =
            Polynomial::with_coefficients(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        p.pad(0);
        assert_eq!(p.len(), 2);
        assert_eq!(
            p.take(),
            vec![Scalar::from_const(12), Scalar::from_const(34)]
        );
    }

    #[test]
    fn test_pad_preserves_evaluation() {
        let mut p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let before = p.evaluate(Scalar::from_const(7));
        p.pad(6);
        assert_eq!(p.evaluate(Scalar::from_const(7)), before);
    }

    #[test]
    fn test_no_roots() {
        let p = from_roots(&[]);
        assert_eq!(p.len(), 1);
        assert_eq!(p.degree_bound(), 1);
        assert_ne!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
    }

    #[test]
    fn test_one_root() {
        let p = from_roots(&[Scalar::from_const(12)]);
        assert_eq!(p.len(), 2);
        assert_eq!(p.degree_bound(), 2);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(12));
        assert_eq!(q.len(), 1);
        assert_eq!(q.degree_bound(), 1);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(34));
        assert_eq!(q.len(), 1);
        assert_eq!(q.degree_bound(), 1);
        assert_ne!(v, Scalar::from_const(0));
    }

    #[test]
    fn test_three_roots() {
        let p = from_roots(&[
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(p.len(), 4);
        assert_eq!(p.degree_bound(), 4);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(12));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = q.horner(Scalar::from_const(34));
        assert_eq!(q.len(), 2);
        assert_eq!(q.degree_bound(), 2);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = q.horner(Scalar::from_const(56));
        assert_eq!(q.len(), 1);
        assert_eq!(q.degree_bound(), 1);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(78));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_ne!(v, Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(90));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_ne!(v, Scalar::from_const(0));
    }

    #[test]
    fn test_three_roots_reverse_order() {
        let p = from_roots(&[
            Scalar::from_const(56),
            Scalar::from_const(34),
            Scalar::from_const(12),
        ]);
        assert_eq!(p.len(), 4);
        assert_eq!(p.degree_bound(), 4);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(12));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = q.horner(Scalar::from_const(34));
        assert_eq!(q.len(), 2);
        assert_eq!(q.degree_bound(), 2);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = q.horner(Scalar::from_const(56));
        assert_eq!(q.len(), 1);
        assert_eq!(q.degree_bound(), 1);
        assert_eq!(v, Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(78));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_ne!(v, Scalar::from_const(0));
        let (q, v) = p.horner(Scalar::from_const(90));
        assert_eq!(q.len(), 3);
        assert_eq!(q.degree_bound(), 3);
        assert_ne!(v, Scalar::from_const(0));
    }

    #[test]
    fn test_seven_roots() {
        let p = from_roots(&[
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
            Scalar::from_const(90),
            Scalar::from_const(13),
            Scalar::from_const(57),
        ]);
        assert_eq!(p.len(), 8);
        assert_eq!(p.degree_bound(), 8);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
    }

    #[test]
    fn test_seven_roots_reverse_order() {
        let p = from_roots(&[
            Scalar::from_const(57),
            Scalar::from_const(13),
            Scalar::from_const(90),
            Scalar::from_const(78),
            Scalar::from_const(56),
            Scalar::from_const(34),
            Scalar::from_const(12),
        ]);
        assert_eq!(p.len(), 8);
        assert_eq!(p.degree_bound(), 8);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(13)), Scalar::from_const(0));
        assert_eq!(p.evaluate(Scalar::from_const(57)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(92)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(46)), Scalar::from_const(0));
        assert_ne!(p.evaluate(Scalar::from_const(80)), Scalar::from_const(0));
    }

    #[test]
    fn test_duplicate_roots() {
        assert!(
            Polynomial::from_roots(
                &[
                    Scalar::from_const(12),
                    Scalar::from_const(34),
                    Scalar::from_const(56),
                    Scalar::from_const(12),
                    Scalar::from_const(90),
                    Scalar::from_const(12),
                    Scalar::from_const(57),
                ],
                get_random_scalar()
            )
            .is_err()
        );
    }

    #[test]
    fn test_interpolate_zero_points() {
        let p = Polynomial::interpolate(&[]).unwrap();
        assert_eq!(p, Polynomial::default());
    }

    #[test]
    fn test_interpolate_one_point1() {
        let p =
            Polynomial::interpolate(&[(Scalar::from_const(12), Scalar::from_const(34))]).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.degree_bound(), 1);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(34));
    }

    #[test]
    fn test_interpolate_one_point2() {
        let p =
            Polynomial::interpolate(&[(Scalar::from_const(34), Scalar::from_const(56))]).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p.degree_bound(), 1);
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(56));
    }

    #[test]
    fn test_interpolate_two_points1() {
        let p = Polynomial::interpolate(&[
            (Scalar::from_const(12), Scalar::from_const(34)),
            (Scalar::from_const(56), Scalar::from_const(78)),
        ])
        .unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p.degree_bound(), 2);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(34));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(78));
    }

    #[test]
    fn test_interpolate_two_points2() {
        let p = Polynomial::interpolate(&[
            (Scalar::from_const(34), Scalar::from_const(12)),
            (Scalar::from_const(78), Scalar::from_const(56)),
        ])
        .unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p.degree_bound(), 2);
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(12));
        assert_eq!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(56));
    }

    #[test]
    fn test_interpolate_three_points1() {
        let p = Polynomial::interpolate(&[
            (Scalar::from_const(12), Scalar::from_const(34)),
            (Scalar::from_const(56), Scalar::from_const(78)),
            (Scalar::from_const(90), Scalar::from_const(12)),
        ])
        .unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p.degree_bound(), 3);
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(34));
        assert_eq!(p.evaluate(Scalar::from_const(56)), Scalar::from_const(78));
        assert_eq!(p.evaluate(Scalar::from_const(90)), Scalar::from_const(12));
    }

    #[test]
    fn test_interpolate_three_points2() {
        let p = Polynomial::interpolate(&[
            (Scalar::from_const(34), Scalar::from_const(12)),
            (Scalar::from_const(78), Scalar::from_const(56)),
            (Scalar::from_const(12), Scalar::from_const(90)),
        ])
        .unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p.degree_bound(), 3);
        assert_eq!(p.evaluate(Scalar::from_const(34)), Scalar::from_const(12));
        assert_eq!(p.evaluate(Scalar::from_const(78)), Scalar::from_const(56));
        assert_eq!(p.evaluate(Scalar::from_const(12)), Scalar::from_const(90));
    }

    #[test]
    fn test_duplicate_coordinates() {
        assert!(
            Polynomial::interpolate(&[
                (Scalar::from_const(12), Scalar::from_const(34)),
                (Scalar::from_const(56), Scalar::from_const(78)),
                (Scalar::from_const(12), Scalar::from_const(90)),
            ])
            .is_err()
        );
    }

    #[test]
    fn test_encode2_one_value_1() {
        let p1 = Polynomial::encode2(vec![Scalar::from_const(42)]);
        let p2 = Polynomial::encode2(vec![Scalar::from_const(42)]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 1);
        assert_eq!(p1.degree_bound(), 1);
        assert_eq!(p2.len(), 1);
        assert_eq!(p2.degree_bound(), 1);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(0, 1)),
            Scalar::from_const(42)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(0, 1), Scalar::from_const(42));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 1)),
            Scalar::from_const(42)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 1), Scalar::from_const(42));
    }

    #[test]
    fn test_encode2_one_value_2() {
        let p1 = Polynomial::encode2(vec![Scalar::from_const(42)]);
        let p2 = Polynomial::encode2(vec![Scalar::from_const(123)]);
        assert_eq!(p2.len(), 1);
        assert_eq!(p2.degree_bound(), 1);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 1)),
            Scalar::from_const(123)
        );
        assert_eq!(
            p2.evaluate_on_two_adic_domain(0, 1),
            Scalar::from_const(123)
        );
    }

    #[test]
    fn test_encode2_two_values_1() {
        let p1 = Polynomial::encode2(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        let p2 = Polynomial::encode2(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 2);
        assert_eq!(p1.degree_bound(), 2);
        assert_eq!(p2.len(), 2);
        assert_eq!(p2.degree_bound(), 2);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(0, 2)),
            Scalar::from_const(12)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(0, 2), Scalar::from_const(12));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(1, 2)),
            Scalar::from_const(34)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(1, 2), Scalar::from_const(34));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 2)),
            Scalar::from_const(12)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 2), Scalar::from_const(12));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 2)),
            Scalar::from_const(34)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 2), Scalar::from_const(34));
    }

    #[test]
    fn test_encode2_two_values_2() {
        let p1 = Polynomial::encode2(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        let p2 = Polynomial::encode2(vec![Scalar::from_const(78), Scalar::from_const(56)]);
        assert_eq!(p1.len(), 2);
        assert_eq!(p1.degree_bound(), 2);
        assert_eq!(p2.len(), 2);
        assert_eq!(p2.degree_bound(), 2);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 2)),
            Scalar::from_const(78)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 2), Scalar::from_const(78));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 2)),
            Scalar::from_const(56)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 2), Scalar::from_const(56));
    }

    #[test]
    fn test_encode2_three_values_1() {
        let p1 = Polynomial::encode2(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        let p2 = Polynomial::encode2(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 4);
        assert_eq!(p1.degree_bound(), 4);
        assert_eq!(p2.len(), 4);
        assert_eq!(p2.degree_bound(), 4);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(0, 3), Scalar::from_const(12));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(0, 4)),
            Scalar::from_const(12)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(0, 4), Scalar::from_const(12));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(1, 3), Scalar::from_const(34));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(1, 4)),
            Scalar::from_const(34)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(1, 4), Scalar::from_const(34));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(2, 3)),
            Scalar::from_const(56)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(2, 3), Scalar::from_const(56));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(2, 4)),
            Scalar::from_const(56)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(2, 4), Scalar::from_const(56));
        assert_eq!(
            p1.evaluate(Polynomial::domain_element2(3, 4)),
            Scalar::from_const(0)
        );
        assert_eq!(p1.evaluate_on_two_adic_domain(3, 4), Scalar::from_const(0));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 3), Scalar::from_const(12));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 4)),
            Scalar::from_const(12)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 4), Scalar::from_const(12));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 3), Scalar::from_const(34));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 4)),
            Scalar::from_const(34)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 4), Scalar::from_const(34));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(2, 3)),
            Scalar::from_const(56)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(2, 3), Scalar::from_const(56));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(2, 4)),
            Scalar::from_const(56)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(2, 4), Scalar::from_const(56));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(3, 4)),
            Scalar::from_const(0)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(3, 4), Scalar::from_const(0));
    }

    #[test]
    fn test_encode2_three_values_2() {
        let p1 = Polynomial::encode2(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        let p2 = Polynomial::encode2(vec![
            Scalar::from_const(90),
            Scalar::from_const(78),
            Scalar::from_const(34),
        ]);
        assert_eq!(p1.len(), 4);
        assert_eq!(p1.degree_bound(), 4);
        assert_eq!(p2.len(), 4);
        assert_eq!(p2.degree_bound(), 4);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 3)),
            Scalar::from_const(90)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 3), Scalar::from_const(90));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(0, 4)),
            Scalar::from_const(90)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(0, 4), Scalar::from_const(90));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 3)),
            Scalar::from_const(78)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 3), Scalar::from_const(78));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(1, 4)),
            Scalar::from_const(78)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(1, 4), Scalar::from_const(78));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(2, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(2, 3), Scalar::from_const(34));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(2, 4)),
            Scalar::from_const(34)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(2, 4), Scalar::from_const(34));
        assert_eq!(
            p2.evaluate(Polynomial::domain_element2(3, 4)),
            Scalar::from_const(0)
        );
        assert_eq!(p2.evaluate_on_two_adic_domain(3, 4), Scalar::from_const(0));
    }

    #[test]
    fn test_encode2_four_values() {
        let p = Polynomial::encode2(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
        ]);
        assert_eq!(p.len(), 4);
        assert_eq!(p.degree_bound(), 4);
        assert_eq!(
            p.evaluate(Polynomial::domain_element2(0, 4)),
            Scalar::from_const(12)
        );
        assert_eq!(p.evaluate_on_two_adic_domain(0, 4), Scalar::from_const(12));
        assert_eq!(
            p.evaluate(Polynomial::domain_element2(1, 4)),
            Scalar::from_const(34)
        );
        assert_eq!(p.evaluate_on_two_adic_domain(1, 4), Scalar::from_const(34));
        assert_eq!(
            p.evaluate(Polynomial::domain_element2(2, 4)),
            Scalar::from_const(56)
        );
        assert_eq!(p.evaluate_on_two_adic_domain(2, 4), Scalar::from_const(56));
        assert_eq!(
            p.evaluate(Polynomial::domain_element2(3, 4)),
            Scalar::from_const(78)
        );
        assert_eq!(p.evaluate_on_two_adic_domain(3, 4), Scalar::from_const(78));
    }

    #[test]
    fn test_decode2_one_value() {
        let values = vec![Scalar::from_const(42)];
        let polynomial = Polynomial::encode2(values.clone());
        assert_eq!(polynomial.decode2(), values);
    }

    #[test]
    fn test_decode2_two_values() {
        let values = vec![Scalar::from_const(12), Scalar::from_const(34)];
        let polynomial = Polynomial::encode2(values.clone());
        assert_eq!(polynomial.decode2(), values);
    }

    #[test]
    fn test_decode2_three_values() {
        let polynomial = Polynomial::encode2(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(
            polynomial.decode2(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56),
                Scalar::from_const(0)
            ]
        );
    }

    #[test]
    fn test_decode2_four_values() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
        ];
        let polynomial = Polynomial::encode2(values.clone());
        assert_eq!(polynomial.decode2(), values);
    }

    #[test]
    fn test_encode3_one_value_1() {
        let p1 = Polynomial::encode3(vec![Scalar::from_const(42)]);
        let p2 = Polynomial::encode3(vec![Scalar::from_const(42)]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 1);
        assert_eq!(p1.degree_bound(), 1);
        assert_eq!(p2.len(), 1);
        assert_eq!(p2.degree_bound(), 1);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(0, 1)),
            Scalar::from_const(42)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(0, 1),
            Scalar::from_const(42)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 1)),
            Scalar::from_const(42)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 1),
            Scalar::from_const(42)
        );
    }

    #[test]
    fn test_encode3_one_value_2() {
        let p1 = Polynomial::encode3(vec![Scalar::from_const(42)]);
        let p2 = Polynomial::encode3(vec![Scalar::from_const(123)]);
        assert_eq!(p2.len(), 1);
        assert_eq!(p2.degree_bound(), 1);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 1)),
            Scalar::from_const(123)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 1),
            Scalar::from_const(123)
        );
    }

    #[test]
    fn test_encode3_two_values_1() {
        let p1 = Polynomial::encode3(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        let p2 = Polynomial::encode3(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 3);
        assert_eq!(p1.degree_bound(), 3);
        assert_eq!(p2.len(), 3);
        assert_eq!(p2.degree_bound(), 3);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(0, 2)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(0, 2),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(0, 3),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(1, 2)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(1, 2),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(1, 3),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(0)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(0)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 2)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 2),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 3),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(1, 2)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(1, 2),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(1, 3),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(0)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(0)
        );
    }

    #[test]
    fn test_encode3_two_values_2() {
        let p1 = Polynomial::encode3(vec![Scalar::from_const(12), Scalar::from_const(34)]);
        let p2 = Polynomial::encode3(vec![Scalar::from_const(78), Scalar::from_const(56)]);
        assert_eq!(p1.len(), 3);
        assert_eq!(p1.degree_bound(), 3);
        assert_eq!(p2.len(), 3);
        assert_eq!(p2.degree_bound(), 3);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 2)),
            Scalar::from_const(78)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 2),
            Scalar::from_const(78)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(1, 2)),
            Scalar::from_const(56)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(1, 2),
            Scalar::from_const(56)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(0)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(0)
        );
    }

    #[test]
    fn test_encode3_three_values_1() {
        let p1 = Polynomial::encode3(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        let p2 = Polynomial::encode3(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), 3);
        assert_eq!(p1.degree_bound(), 3);
        assert_eq!(p2.len(), 3);
        assert_eq!(p2.degree_bound(), 3);
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(0, 3),
            Scalar::from_const(12)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(1, 3),
            Scalar::from_const(34)
        );
        assert_eq!(
            p1.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(56)
        );
        assert_eq!(
            p1.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(56)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 3)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 3),
            Scalar::from_const(12)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(1, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(1, 3),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(56)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(56)
        );
    }

    #[test]
    fn test_encode3_three_values_2() {
        let p1 = Polynomial::encode3(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ]);
        let p2 = Polynomial::encode3(vec![
            Scalar::from_const(90),
            Scalar::from_const(78),
            Scalar::from_const(34),
        ]);
        assert_eq!(p1.len(), 3);
        assert_eq!(p1.degree_bound(), 3);
        assert_eq!(p2.len(), 3);
        assert_eq!(p2.degree_bound(), 3);
        assert_ne!(p1, p2);
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(0, 3)),
            Scalar::from_const(90)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(0, 3),
            Scalar::from_const(90)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(1, 3)),
            Scalar::from_const(78)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(1, 3),
            Scalar::from_const(78)
        );
        assert_eq!(
            p2.evaluate(Polynomial::domain_element3(2, 3)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p2.evaluate_on_three_adic_domain(2, 3),
            Scalar::from_const(34)
        );
    }

    #[test]
    fn test_encode3_nine_values3() {
        let p = Polynomial::encode3(vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
            Scalar::from_const(90),
            Scalar::from_const(11),
            Scalar::from_const(22),
            Scalar::from_const(33),
            Scalar::from_const(44),
        ]);
        assert_eq!(p.len(), 9);
        assert_eq!(p.degree_bound(), 9);
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(0, 9)),
            Scalar::from_const(12)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(0, 9),
            Scalar::from_const(12)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(1, 9)),
            Scalar::from_const(34)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(1, 9),
            Scalar::from_const(34)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(2, 9)),
            Scalar::from_const(56)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(2, 9),
            Scalar::from_const(56)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(3, 9)),
            Scalar::from_const(78)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(3, 9),
            Scalar::from_const(78)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(4, 9)),
            Scalar::from_const(90)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(4, 9),
            Scalar::from_const(90)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(5, 9)),
            Scalar::from_const(11)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(5, 9),
            Scalar::from_const(11)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(6, 9)),
            Scalar::from_const(22)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(6, 9),
            Scalar::from_const(22)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(7, 9)),
            Scalar::from_const(33)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(7, 9),
            Scalar::from_const(33)
        );
        assert_eq!(
            p.evaluate(Polynomial::domain_element3(8, 9)),
            Scalar::from_const(44)
        );
        assert_eq!(
            p.evaluate_on_three_adic_domain(8, 9),
            Scalar::from_const(44)
        );
    }

    #[test]
    fn test_decode3_one_value() {
        let values = vec![Scalar::from_const(42)];
        let polynomial = Polynomial::encode3(values.clone());
        assert_eq!(polynomial.decode3(), values);
    }

    #[test]
    fn test_decode3_two_values() {
        let values = vec![Scalar::from_const(12), Scalar::from_const(34)];
        let polynomial = Polynomial::encode3(values.clone());
        assert_eq!(
            polynomial.decode3(),
            vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(0)
            ]
        );
    }

    #[test]
    fn test_decode3_three_values() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ];
        let polynomial = Polynomial::encode3(values.clone());
        assert_eq!(polynomial.decode3(), values);
    }

    #[test]
    fn test_decode3_nine_values() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
            Scalar::from_const(90),
            Scalar::from_const(11),
            Scalar::from_const(22),
            Scalar::from_const(33),
            Scalar::from_const(44),
        ];
        let polynomial = Polynomial::encode3(values.clone());
        assert_eq!(polynomial.decode3(), values);
    }

    #[test]
    fn test_add_same_length() {
        let p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        assert_eq!(
            p1 + p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(33)
            ])
        );
    }

    #[test]
    fn test_add_lhs_longer() {
        let p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let p2 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        assert_eq!(
            p1 + p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(3)
            ])
        );
    }

    #[test]
    fn test_add_rhs_longer() {
        let p1 = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        assert_eq!(
            p1 + p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(30)
            ])
        );
    }

    #[test]
    fn test_add_commutative() {
        let p1 = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        assert_eq!(p1.clone() + p2.clone(), p2 + p1);
    }

    #[test]
    fn test_add_assign_same_length() {
        let mut p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        p1 += p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(33)
            ])
        );
    }

    #[test]
    fn test_add_assign_lhs_longer() {
        let mut p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let p2 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        p1 += p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(3)
            ])
        );
    }

    #[test]
    fn test_add_assign_rhs_longer() {
        let mut p1 =
            Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        p1 += p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(11),
                Scalar::from_const(22),
                Scalar::from_const(30)
            ])
        );
    }

    #[test]
    fn test_add_assign_consistent_with_add() {
        let p1 = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        let mut p1_assign = p1.clone();
        p1_assign += p2.clone();
        assert_eq!(p1_assign, p1 + p2);
    }

    #[test]
    fn test_sub_same_length() {
        let p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        assert_eq!(
            p1 - p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                Scalar::from_const(27)
            ])
        );
    }

    #[test]
    fn test_sub_lhs_longer() {
        let p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        let p2 = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        assert_eq!(
            p1 - p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                Scalar::from_const(30)
            ])
        );
    }

    #[test]
    fn test_sub_rhs_longer() {
        let p1 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        assert_eq!(
            p1 - p2,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                -Scalar::from_const(3)
            ])
        );
    }

    #[test]
    fn test_sub_anticommutative() {
        let p1 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        assert_eq!(p1.clone() - p2.clone(), -(p2 - p1));
    }

    #[test]
    fn test_sub_assign_same_length() {
        let mut p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        p1 -= p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                Scalar::from_const(27)
            ])
        );
    }

    #[test]
    fn test_sub_assign_lhs_longer() {
        let mut p1 = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(20),
            Scalar::from_const(30),
        ]);
        let p2 = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        p1 -= p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                Scalar::from_const(30)
            ])
        );
    }

    #[test]
    fn test_sub_assign_rhs_longer() {
        let mut p1 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        p1 -= p2;
        assert_eq!(
            p1,
            Polynomial::with_coefficients(vec![
                Scalar::from_const(9),
                Scalar::from_const(18),
                -Scalar::from_const(3)
            ])
        );
    }

    #[test]
    fn test_sub_assign_consistent_with_sub() {
        let p1 =
            Polynomial::with_coefficients(vec![Scalar::from_const(10), Scalar::from_const(20)]);
        let p2 = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let mut p1_assign = p1.clone();
        p1_assign -= p2.clone();
        assert_eq!(p1_assign, p1 - p2);
    }

    #[test]
    fn test_multiply_empty() {
        let p1 = Polynomial::default();
        let p2 = Polynomial::default();
        assert_eq!(p1.multiply(p2), Polynomial::default());
    }

    #[test]
    fn test_multiply_empty_by_non_empty() {
        let p1 = Polynomial::default();
        let p2 = Polynomial {
            coefficients: vec![Scalar::from_const(12), Scalar::from_const(34)],
        };
        assert_eq!(p1.multiply(p2), Polynomial::default());
    }

    #[test]
    fn test_multiply_non_empty_by_empty() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(56), Scalar::from_const(78)],
        };
        let p2 = Polynomial::default();
        assert_eq!(p1.multiply(p2), Polynomial::default());
    }

    #[test]
    fn test_multiply_constant() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(3)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56),
            ],
        };
        assert_eq!(
            p1.multiply(p2),
            Polynomial {
                coefficients: vec![
                    Scalar::from_const(36),
                    Scalar::from_const(102),
                    Scalar::from_const(168)
                ]
            }
        );
    }

    #[test]
    fn test_multiply_by_constant() {
        let p1 = Polynomial {
            coefficients: vec![
                Scalar::from_const(12),
                Scalar::from_const(34),
                Scalar::from_const(56),
            ],
        };
        let p2 = Polynomial {
            coefficients: vec![Scalar::from_const(3)],
        };
        assert_eq!(
            p1.multiply(p2),
            Polynomial {
                coefficients: vec![
                    Scalar::from_const(36),
                    Scalar::from_const(102),
                    Scalar::from_const(168)
                ]
            }
        );
    }

    #[test]
    fn test_multiply_constant_by_constant() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(12)],
        };
        let p2 = Polynomial {
            coefficients: vec![Scalar::from_const(34)],
        };
        assert_eq!(
            p1.multiply(p2),
            Polynomial {
                coefficients: vec![Scalar::from_const(408)]
            }
        );
    }

    #[test]
    fn test_multiply_polynomials1() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![Scalar::from_const(3), Scalar::from_const(4)],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(10),
                Scalar::from_const(8),
            ],
        };
        assert_eq!(p1.clone().multiply(p2.clone()), result);
        assert_eq!(p2.multiply(p1), result);
    }

    #[test]
    fn test_multiply_polynomials2() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(4),
                Scalar::from_const(5),
            ],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(10),
                Scalar::from_const(13),
                Scalar::from_const(10),
            ],
        };
        assert_eq!(p1.clone().multiply(p2.clone()), result);
        assert_eq!(p2.multiply(p1), result);
    }

    #[test]
    fn test_polynomial_mul_op() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(4),
                Scalar::from_const(5),
            ],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(10),
                Scalar::from_const(13),
                Scalar::from_const(10),
            ],
        };
        assert_eq!(p1.clone() * p2.clone(), result);
        assert_eq!(p2 * p1, result);
    }

    #[test]
    fn test_polynomial_mul_assign() {
        let mut p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(4),
                Scalar::from_const(5),
            ],
        };
        p1 *= p2;
        assert_eq!(
            p1,
            Polynomial {
                coefficients: vec![
                    Scalar::from_const(3),
                    Scalar::from_const(10),
                    Scalar::from_const(13),
                    Scalar::from_const(10)
                ],
            }
        );
    }

    #[test]
    fn test_multiply_one_polynomial() {
        let p = Polynomial {
            coefficients: vec![Scalar::from_const(12), Scalar::from_const(34)],
        };
        assert_eq!(Polynomial::multiply_many([p.clone()]), p);
    }

    #[test]
    fn test_multiply_two_polynomials() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(4),
                Scalar::from_const(5),
            ],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(10),
                Scalar::from_const(13),
                Scalar::from_const(10),
            ],
        };
        assert_eq!(Polynomial::multiply_many([p1.clone(), p2.clone()]), result);
        assert_eq!(Polynomial::multiply_many([p2, p1]), result);
    }

    #[test]
    fn test_multiply_three_polynomials() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![
                Scalar::from_const(3),
                Scalar::from_const(4),
                Scalar::from_const(5),
            ],
        };
        let p3 = Polynomial {
            coefficients: vec![
                Scalar::from_const(6),
                Scalar::from_const(7),
                Scalar::from_const(8),
                Scalar::from_const(9),
            ],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(18),
                Scalar::from_const(81),
                Scalar::from_const(172),
                Scalar::from_const(258),
                Scalar::from_const(264),
                Scalar::from_const(197),
                Scalar::from_const(90),
            ],
        };
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p2.clone(), p3.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p3.clone(), p2.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p2.clone(), p1.clone(), p3.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p2.clone(), p3.clone(), p1.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p3.clone(), p1.clone(), p2.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p3.clone(), p2.clone(), p1.clone()]),
            result
        );
    }

    #[test]
    fn test_multiply_four_polynomials() {
        let p1 = Polynomial {
            coefficients: vec![Scalar::from_const(1), Scalar::from_const(2)],
        };
        let p2 = Polynomial {
            coefficients: vec![Scalar::from_const(3), Scalar::from_const(4)],
        };
        let p3 = Polynomial {
            coefficients: vec![Scalar::from_const(5), Scalar::from_const(6)],
        };
        let p4 = Polynomial {
            coefficients: vec![Scalar::from_const(7), Scalar::from_const(8)],
        };
        let result = Polynomial {
            coefficients: vec![
                Scalar::from_const(105),
                Scalar::from_const(596),
                Scalar::from_const(1244),
                Scalar::from_const(1136),
                Scalar::from_const(384),
            ],
        };
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p2.clone(), p3.clone(), p4.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p2.clone(), p4.clone(), p3.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p3.clone(), p2.clone(), p4.clone()]),
            result
        );
        assert_eq!(
            Polynomial::multiply_many([p1.clone(), p3.clone(), p4.clone(), p2.clone()]),
            result
        );
        // okay, not gonna try all permutations -- too much typing for too little gain.
    }

    #[test]
    fn test_divide_zero_by_zero() {
        let z = Polynomial {
            coefficients: vec![
                -Scalar::from_const(1),
                Scalar::from_const(0),
                Scalar::from_const(0),
                Scalar::from_const(0),
                Scalar::from_const(1),
            ],
        };
        assert_eq!(
            z.divide_by_zero(4).unwrap(),
            Polynomial {
                coefficients: vec![Scalar::from_const(1)]
            }
        );
    }

    #[test]
    fn test_non_trivial_quotient1() {
        let ql = Polynomial::encode2(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(1),
            Scalar::from_const(1),
        ]);
        let qr = Polynomial::encode2(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(1),
            Scalar::from_const(1),
        ]);
        let qo = Polynomial::encode2(vec![-Scalar::from_const(1); 4]);
        let qm = Polynomial::encode2(vec![
            Scalar::from_const(1),
            Scalar::from_const(1),
            Scalar::from_const(0),
            Scalar::from_const(0),
        ]);
        let qc = Polynomial::encode2(vec![Scalar::from_const(0); 4]);
        let l = Polynomial::encode2(vec![
            Scalar::from_const(3),
            Scalar::from_const(9),
            Scalar::from_const(3),
            Scalar::from_const(30),
        ]);
        let r = Polynomial::encode2(vec![
            Scalar::from_const(3),
            Scalar::from_const(3),
            Scalar::from_const(27),
            Scalar::from_const(5),
        ]);
        let o = Polynomial::encode2(vec![
            Scalar::from_const(9),
            Scalar::from_const(27),
            Scalar::from_const(30),
            Scalar::from_const(35),
        ]);
        let lr = l.clone().multiply(r.clone());
        let p = ql.multiply(l) + qr.multiply(r) + qo.multiply(o) + qm.multiply(lr) + qc;
        let q = p.divide_by_zero(4).unwrap();
        assert_eq!(q.len(), 6);
        assert_eq!(q.degree_bound(), 6);
    }

    #[test]
    fn test_non_trivial_quotient2() {
        let ql = Polynomial::encode2(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(1),
            Scalar::from_const(1),
        ]);
        let qr = Polynomial::encode2(vec![
            Scalar::from_const(0),
            Scalar::from_const(0),
            Scalar::from_const(1),
            Scalar::from_const(5),
        ]);
        let qo = Polynomial::encode2(vec![-Scalar::from_const(1); 4]);
        let qm = Polynomial::encode2(vec![
            Scalar::from_const(1),
            Scalar::from_const(1),
            Scalar::from_const(0),
            Scalar::from_const(0),
        ]);
        let qc = Polynomial::encode2(vec![Scalar::from_const(0); 4]);
        let l = Polynomial::encode2(vec![
            Scalar::from_const(3),
            Scalar::from_const(9),
            Scalar::from_const(3),
            Scalar::from_const(30),
        ]);
        let r = Polynomial::encode2(vec![
            Scalar::from_const(3),
            Scalar::from_const(3),
            Scalar::from_const(27),
            Scalar::from_const(1),
        ]);
        let o = Polynomial::encode2(vec![
            Scalar::from_const(9),
            Scalar::from_const(27),
            Scalar::from_const(30),
            Scalar::from_const(35),
        ]);
        let lr = l.clone().multiply(r.clone());
        let p = ql.multiply(l) + qr.multiply(r) + qo.multiply(o) + qm.multiply(lr) + qc;
        let q = p.divide_by_zero(4).unwrap();
        assert_eq!(q.len(), 6);
        assert_eq!(q.degree_bound(), 6);
    }

    #[test]
    fn test_lde2_same_size() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
        ];
        let p = Polynomial::encode2(values);
        let lde = p.clone().shifted_lde2(4);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_two_adic_coset(0, 4),
                p.evaluate_on_two_adic_coset(1, 4),
                p.evaluate_on_two_adic_coset(2, 4),
                p.evaluate_on_two_adic_coset(3, 4),
            ]
        );
    }

    #[test]
    fn test_lde2_blowup2() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
            Scalar::from_const(78),
        ];
        let p = Polynomial::encode2(values);
        let lde = p.clone().shifted_lde2(8);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_two_adic_coset(0, 8),
                p.evaluate_on_two_adic_coset(1, 8),
                p.evaluate_on_two_adic_coset(2, 8),
                p.evaluate_on_two_adic_coset(3, 8),
                p.evaluate_on_two_adic_coset(4, 8),
                p.evaluate_on_two_adic_coset(5, 8),
                p.evaluate_on_two_adic_coset(6, 8),
                p.evaluate_on_two_adic_coset(7, 8),
            ]
        );
    }

    #[test]
    fn test_lde2_blowup4() {
        let values = vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ];
        let p = Polynomial::encode2(values);
        let lde = p.clone().shifted_lde2(16);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_two_adic_coset(0, 16),
                p.evaluate_on_two_adic_coset(1, 16),
                p.evaluate_on_two_adic_coset(2, 16),
                p.evaluate_on_two_adic_coset(3, 16),
                p.evaluate_on_two_adic_coset(4, 16),
                p.evaluate_on_two_adic_coset(5, 16),
                p.evaluate_on_two_adic_coset(6, 16),
                p.evaluate_on_two_adic_coset(7, 16),
                p.evaluate_on_two_adic_coset(8, 16),
                p.evaluate_on_two_adic_coset(9, 16),
                p.evaluate_on_two_adic_coset(10, 16),
                p.evaluate_on_two_adic_coset(11, 16),
                p.evaluate_on_two_adic_coset(12, 16),
                p.evaluate_on_two_adic_coset(13, 16),
                p.evaluate_on_two_adic_coset(14, 16),
                p.evaluate_on_two_adic_coset(15, 16),
            ]
        );
    }

    #[test]
    fn test_lde2_shorter_polynomial() {
        let values = vec![Scalar::from_const(42), Scalar::from_const(42)];
        let p = Polynomial::encode2(values);
        assert_eq!(p.len(), 1);
        assert_eq!(p.degree_bound(), 1);
        let lde = p.clone().shifted_lde2(4);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_two_adic_coset(0, 4),
                p.evaluate_on_two_adic_coset(1, 4),
                p.evaluate_on_two_adic_coset(2, 4),
                p.evaluate_on_two_adic_coset(3, 4),
            ]
        );
    }

    #[test]
    fn test_lde3_same_size() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ];
        let p = Polynomial::encode3(values.clone());
        let lde = p.clone().shifted_lde3(3);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_three_adic_coset(0, 3),
                p.evaluate_on_three_adic_coset(1, 3),
                p.evaluate_on_three_adic_coset(2, 3),
            ]
        );
    }

    #[test]
    fn test_lde3_blowup3() {
        let values = vec![
            Scalar::from_const(12),
            Scalar::from_const(34),
            Scalar::from_const(56),
        ];
        let p = Polynomial::encode3(values);
        let lde = p.clone().shifted_lde3(9);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_three_adic_coset(0, 9),
                p.evaluate_on_three_adic_coset(1, 9),
                p.evaluate_on_three_adic_coset(2, 9),
                p.evaluate_on_three_adic_coset(3, 9),
                p.evaluate_on_three_adic_coset(4, 9),
                p.evaluate_on_three_adic_coset(5, 9),
                p.evaluate_on_three_adic_coset(6, 9),
                p.evaluate_on_three_adic_coset(7, 9),
                p.evaluate_on_three_adic_coset(8, 9),
            ]
        );
    }

    #[test]
    fn test_lde3_blowup9() {
        let values = vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ];
        let p = Polynomial::encode3(values);
        let lde = p.clone().shifted_lde3(27);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_three_adic_coset(0, 27),
                p.evaluate_on_three_adic_coset(1, 27),
                p.evaluate_on_three_adic_coset(2, 27),
                p.evaluate_on_three_adic_coset(3, 27),
                p.evaluate_on_three_adic_coset(4, 27),
                p.evaluate_on_three_adic_coset(5, 27),
                p.evaluate_on_three_adic_coset(6, 27),
                p.evaluate_on_three_adic_coset(7, 27),
                p.evaluate_on_three_adic_coset(8, 27),
                p.evaluate_on_three_adic_coset(9, 27),
                p.evaluate_on_three_adic_coset(10, 27),
                p.evaluate_on_three_adic_coset(11, 27),
                p.evaluate_on_three_adic_coset(12, 27),
                p.evaluate_on_three_adic_coset(13, 27),
                p.evaluate_on_three_adic_coset(14, 27),
                p.evaluate_on_three_adic_coset(15, 27),
                p.evaluate_on_three_adic_coset(16, 27),
                p.evaluate_on_three_adic_coset(17, 27),
                p.evaluate_on_three_adic_coset(18, 27),
                p.evaluate_on_three_adic_coset(19, 27),
                p.evaluate_on_three_adic_coset(20, 27),
                p.evaluate_on_three_adic_coset(21, 27),
                p.evaluate_on_three_adic_coset(22, 27),
                p.evaluate_on_three_adic_coset(23, 27),
                p.evaluate_on_three_adic_coset(24, 27),
                p.evaluate_on_three_adic_coset(25, 27),
                p.evaluate_on_three_adic_coset(26, 27),
            ]
        );
    }

    #[test]
    fn test_lde3_nine_values_blowup3() {
        let values = (1u64..=9).map(Scalar::from).collect();
        let p = Polynomial::encode3(values);
        let lde = p.clone().shifted_lde3(27);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_three_adic_coset(0, 27),
                p.evaluate_on_three_adic_coset(1, 27),
                p.evaluate_on_three_adic_coset(2, 27),
                p.evaluate_on_three_adic_coset(3, 27),
                p.evaluate_on_three_adic_coset(4, 27),
                p.evaluate_on_three_adic_coset(5, 27),
                p.evaluate_on_three_adic_coset(6, 27),
                p.evaluate_on_three_adic_coset(7, 27),
                p.evaluate_on_three_adic_coset(8, 27),
                p.evaluate_on_three_adic_coset(9, 27),
                p.evaluate_on_three_adic_coset(10, 27),
                p.evaluate_on_three_adic_coset(11, 27),
                p.evaluate_on_three_adic_coset(12, 27),
                p.evaluate_on_three_adic_coset(13, 27),
                p.evaluate_on_three_adic_coset(14, 27),
                p.evaluate_on_three_adic_coset(15, 27),
                p.evaluate_on_three_adic_coset(16, 27),
                p.evaluate_on_three_adic_coset(17, 27),
                p.evaluate_on_three_adic_coset(18, 27),
                p.evaluate_on_three_adic_coset(19, 27),
                p.evaluate_on_three_adic_coset(20, 27),
                p.evaluate_on_three_adic_coset(21, 27),
                p.evaluate_on_three_adic_coset(22, 27),
                p.evaluate_on_three_adic_coset(23, 27),
                p.evaluate_on_three_adic_coset(24, 27),
                p.evaluate_on_three_adic_coset(25, 27),
                p.evaluate_on_three_adic_coset(26, 27),
            ]
        );
    }

    #[test]
    fn test_lde3_shorter_poly() {
        let values = vec![
            Scalar::from_const(7),
            Scalar::from_const(7),
            Scalar::from_const(7),
        ];
        let p = Polynomial::encode3(values);
        assert_eq!(p.len(), 1);
        assert_eq!(p.degree_bound(), 1);
        let lde = p.clone().shifted_lde3(9);
        assert_eq!(
            lde,
            vec![
                p.evaluate_on_three_adic_domain(0, 9),
                p.evaluate_on_three_adic_domain(1, 9),
                p.evaluate_on_three_adic_domain(2, 9),
                p.evaluate_on_three_adic_domain(3, 9),
                p.evaluate_on_three_adic_domain(4, 9),
                p.evaluate_on_three_adic_domain(5, 9),
                p.evaluate_on_three_adic_domain(6, 9),
                p.evaluate_on_three_adic_domain(7, 9),
                p.evaluate_on_three_adic_domain(8, 9),
            ]
        );
    }

    #[test]
    fn test_fold2_degree_zero() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(5)]);
        assert_eq!(
            p.clone().fold2(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(5)]
        );
        assert_eq!(
            p.fold2(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(5)]
        );
    }

    #[test]
    fn test_fold2_degree_one() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(2), Scalar::from_const(3)]);
        assert_eq!(
            p.clone().fold2(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(8)]
        );
        assert_eq!(
            p.fold2(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(11)]
        );
    }

    #[test]
    fn test_fold2_degree_two() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        assert_eq!(
            p.clone().fold2(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(5), Scalar::from_const(3)],
        );
        assert_eq!(
            p.fold2(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(7), Scalar::from_const(3)],
        );
    }

    #[test]
    fn test_fold2_degree_three() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ]);
        assert_eq!(
            p.clone().fold2(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(5), Scalar::from_const(11)],
        );
        assert_eq!(
            p.fold2(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(7), Scalar::from_const(15)],
        );
    }

    #[test]
    fn test_fold3_degree_zero() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(5)]);
        assert_eq!(
            p.clone().fold3(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(5)]
        );
        assert_eq!(
            p.fold3(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(5)]
        );
    }

    #[test]
    fn test_fold3_degree_two() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        assert_eq!(
            p.clone().fold3(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(17)]
        );
        assert_eq!(
            p.fold3(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(34)]
        );
    }

    #[test]
    fn test_fold3_degree_three() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ]);
        assert_eq!(
            p.clone().fold3(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(17), Scalar::from_const(4)],
        );
        assert_eq!(
            p.fold3(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(34), Scalar::from_const(4)],
        );
    }

    #[test]
    fn test_fold3_degree_five() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
            Scalar::from_const(5),
            Scalar::from_const(6),
        ]);
        assert_eq!(
            p.clone().fold3(Scalar::from_const(2)).take(),
            vec![Scalar::from_const(17), Scalar::from_const(38)],
        );
        assert_eq!(
            p.fold3(Scalar::from_const(3)).take(),
            vec![Scalar::from_const(34), Scalar::from_const(73)],
        );
    }

    #[test]
    fn test_multiply_values2_same_constant() {
        let lhs = vec![Scalar::from_const(42), Scalar::from_const(42)];
        let rhs = vec![Scalar::from_const(42), Scalar::from_const(42)];
        let result = Polynomial::multiply_values2(lhs, rhs);
        assert_eq!(result, vec![Scalar::from_const(1764)]);
    }

    #[test]
    fn test_multiply_values2_different_constants() {
        let lhs = vec![Scalar::from_const(3), Scalar::from_const(3)];
        let rhs = vec![Scalar::from_const(7), Scalar::from_const(7)];
        let result = Polynomial::multiply_values2(lhs, rhs);
        assert_eq!(result, vec![Scalar::from_const(21)]);
    }

    #[test]
    fn test_multiply_values2_two_linear_polynomials() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let q = Polynomial::with_coefficients(vec![Scalar::from_const(3), Scalar::from_const(4)]);
        let lhs = vec![
            p.evaluate_on_two_adic_domain(0, 2),
            p.evaluate_on_two_adic_domain(1, 2),
        ];
        let rhs = vec![
            q.evaluate_on_two_adic_domain(0, 2),
            q.evaluate_on_two_adic_domain(1, 2),
        ];
        let product = p.multiply(q);
        let result = Polynomial::multiply_values2(lhs, rhs);
        assert_eq!(
            result,
            vec![
                product.evaluate_on_two_adic_domain(0, 4),
                product.evaluate_on_two_adic_domain(1, 4),
                product.evaluate_on_two_adic_domain(2, 4),
                product.evaluate_on_two_adic_domain(3, 4),
            ]
        );
    }

    #[test]
    fn test_multiply_values2_four_values() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ]);
        let q = Polynomial::with_coefficients(vec![
            Scalar::from_const(5),
            Scalar::from_const(6),
            Scalar::from_const(7),
            Scalar::from_const(8),
        ]);
        let lhs = vec![
            p.evaluate_on_two_adic_domain(0, 4),
            p.evaluate_on_two_adic_domain(1, 4),
            p.evaluate_on_two_adic_domain(2, 4),
            p.evaluate_on_two_adic_domain(3, 4),
        ];
        let rhs = vec![
            q.evaluate_on_two_adic_domain(0, 4),
            q.evaluate_on_two_adic_domain(1, 4),
            q.evaluate_on_two_adic_domain(2, 4),
            q.evaluate_on_two_adic_domain(3, 4),
        ];
        let product = p.multiply(q);
        let result = Polynomial::multiply_values2(lhs, rhs);
        assert_eq!(
            result,
            vec![
                product.evaluate_on_two_adic_domain(0, 8),
                product.evaluate_on_two_adic_domain(1, 8),
                product.evaluate_on_two_adic_domain(2, 8),
                product.evaluate_on_two_adic_domain(3, 8),
                product.evaluate_on_two_adic_domain(4, 8),
                product.evaluate_on_two_adic_domain(5, 8),
                product.evaluate_on_two_adic_domain(6, 8),
                product.evaluate_on_two_adic_domain(7, 8),
            ]
        );
    }

    #[test]
    fn test_multiply_values2_commutative() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let q = Polynomial::with_coefficients(vec![Scalar::from_const(3), Scalar::from_const(4)]);
        let values_p = vec![
            p.evaluate_on_two_adic_domain(0, 2),
            p.evaluate_on_two_adic_domain(1, 2),
        ];
        let values_q = vec![
            q.evaluate_on_two_adic_domain(0, 2),
            q.evaluate_on_two_adic_domain(1, 2),
        ];
        let result_pq = Polynomial::multiply_values2(values_p.clone(), values_q.clone());
        let result_qp = Polynomial::multiply_values2(values_q, values_p);
        assert_eq!(result_pq, result_qp);
    }

    #[test]
    fn test_multiply_values2_round_trip() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
        ]);
        let q = Polynomial::with_coefficients(vec![
            Scalar::from_const(5),
            Scalar::from_const(6),
            Scalar::from_const(7),
            Scalar::from_const(8),
        ]);
        let lhs = vec![
            p.evaluate_on_two_adic_domain(0, 4),
            p.evaluate_on_two_adic_domain(1, 4),
            p.evaluate_on_two_adic_domain(2, 4),
            p.evaluate_on_two_adic_domain(3, 4),
        ];
        let rhs = vec![
            q.evaluate_on_two_adic_domain(0, 4),
            q.evaluate_on_two_adic_domain(1, 4),
            q.evaluate_on_two_adic_domain(2, 4),
            q.evaluate_on_two_adic_domain(3, 4),
        ];
        let product = p.clone().multiply(q.clone());
        let result = Polynomial::encode2(Polynomial::multiply_values2(lhs, rhs));
        assert_eq!(result, product);
    }

    #[test]
    fn test_multiply_values3_same_constant() {
        let lhs = vec![
            Scalar::from_const(42),
            Scalar::from_const(42),
            Scalar::from_const(42),
        ];
        let rhs = vec![
            Scalar::from_const(42),
            Scalar::from_const(42),
            Scalar::from_const(42),
        ];
        let result = Polynomial::multiply_values3(lhs, rhs);
        assert_eq!(result, vec![Scalar::from_const(1764)]);
    }

    #[test]
    fn test_multiply_values3_different_constants() {
        let lhs = vec![
            Scalar::from_const(3),
            Scalar::from_const(3),
            Scalar::from_const(3),
        ];
        let rhs = vec![
            Scalar::from_const(7),
            Scalar::from_const(7),
            Scalar::from_const(7),
        ];
        let result = Polynomial::multiply_values3(lhs, rhs);
        assert_eq!(result, vec![Scalar::from_const(21)]);
    }

    #[test]
    fn test_multiply_values3_two_linear_polynomials() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let q = Polynomial::with_coefficients(vec![Scalar::from_const(3), Scalar::from_const(4)]);
        let lhs = vec![
            p.evaluate_on_three_adic_domain(0, 3),
            p.evaluate_on_three_adic_domain(1, 3),
            p.evaluate_on_three_adic_domain(2, 3),
        ];
        let rhs = vec![
            q.evaluate_on_three_adic_domain(0, 3),
            q.evaluate_on_three_adic_domain(1, 3),
            q.evaluate_on_three_adic_domain(2, 3),
        ];
        let product = p.multiply(q);
        let result = Polynomial::multiply_values3(lhs, rhs);
        assert_eq!(
            result,
            vec![
                product.evaluate_on_three_adic_domain(0, 3),
                product.evaluate_on_three_adic_domain(1, 3),
                product.evaluate_on_three_adic_domain(2, 3),
            ]
        );
    }

    #[test]
    fn test_multiply_values3_nine_values() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
            Scalar::from_const(4),
            Scalar::from_const(5),
            Scalar::from_const(6),
            Scalar::from_const(7),
            Scalar::from_const(8),
            Scalar::from_const(9),
        ]);
        let q = Polynomial::with_coefficients(vec![
            Scalar::from_const(10),
            Scalar::from_const(11),
            Scalar::from_const(12),
            Scalar::from_const(13),
            Scalar::from_const(14),
            Scalar::from_const(15),
            Scalar::from_const(16),
            Scalar::from_const(17),
            Scalar::from_const(18),
        ]);
        let lhs = vec![
            p.evaluate_on_three_adic_domain(0, 9),
            p.evaluate_on_three_adic_domain(1, 9),
            p.evaluate_on_three_adic_domain(2, 9),
            p.evaluate_on_three_adic_domain(3, 9),
            p.evaluate_on_three_adic_domain(4, 9),
            p.evaluate_on_three_adic_domain(5, 9),
            p.evaluate_on_three_adic_domain(6, 9),
            p.evaluate_on_three_adic_domain(7, 9),
            p.evaluate_on_three_adic_domain(8, 9),
        ];
        let rhs = vec![
            q.evaluate_on_three_adic_domain(0, 9),
            q.evaluate_on_three_adic_domain(1, 9),
            q.evaluate_on_three_adic_domain(2, 9),
            q.evaluate_on_three_adic_domain(3, 9),
            q.evaluate_on_three_adic_domain(4, 9),
            q.evaluate_on_three_adic_domain(5, 9),
            q.evaluate_on_three_adic_domain(6, 9),
            q.evaluate_on_three_adic_domain(7, 9),
            q.evaluate_on_three_adic_domain(8, 9),
        ];
        let product = p.multiply(q);
        let result = Polynomial::multiply_values3(lhs, rhs);
        assert_eq!(
            result,
            vec![
                product.evaluate_on_three_adic_domain(0, 27),
                product.evaluate_on_three_adic_domain(1, 27),
                product.evaluate_on_three_adic_domain(2, 27),
                product.evaluate_on_three_adic_domain(3, 27),
                product.evaluate_on_three_adic_domain(4, 27),
                product.evaluate_on_three_adic_domain(5, 27),
                product.evaluate_on_three_adic_domain(6, 27),
                product.evaluate_on_three_adic_domain(7, 27),
                product.evaluate_on_three_adic_domain(8, 27),
                product.evaluate_on_three_adic_domain(9, 27),
                product.evaluate_on_three_adic_domain(10, 27),
                product.evaluate_on_three_adic_domain(11, 27),
                product.evaluate_on_three_adic_domain(12, 27),
                product.evaluate_on_three_adic_domain(13, 27),
                product.evaluate_on_three_adic_domain(14, 27),
                product.evaluate_on_three_adic_domain(15, 27),
                product.evaluate_on_three_adic_domain(16, 27),
                product.evaluate_on_three_adic_domain(17, 27),
                product.evaluate_on_three_adic_domain(18, 27),
                product.evaluate_on_three_adic_domain(19, 27),
                product.evaluate_on_three_adic_domain(20, 27),
                product.evaluate_on_three_adic_domain(21, 27),
                product.evaluate_on_three_adic_domain(22, 27),
                product.evaluate_on_three_adic_domain(23, 27),
                product.evaluate_on_three_adic_domain(24, 27),
                product.evaluate_on_three_adic_domain(25, 27),
                product.evaluate_on_three_adic_domain(26, 27),
            ]
        );
    }

    #[test]
    fn test_multiply_values3_commutative() {
        let p = Polynomial::with_coefficients(vec![Scalar::from_const(1), Scalar::from_const(2)]);
        let q = Polynomial::with_coefficients(vec![Scalar::from_const(3), Scalar::from_const(4)]);
        let values_p = vec![
            p.evaluate_on_three_adic_domain(0, 3),
            p.evaluate_on_three_adic_domain(1, 3),
            p.evaluate_on_three_adic_domain(2, 3),
        ];
        let values_q = vec![
            q.evaluate_on_three_adic_domain(0, 3),
            q.evaluate_on_three_adic_domain(1, 3),
            q.evaluate_on_three_adic_domain(2, 3),
        ];
        let result_pq = Polynomial::multiply_values3(values_p.clone(), values_q.clone());
        let result_qp = Polynomial::multiply_values3(values_q, values_p);
        assert_eq!(result_pq, result_qp);
    }

    #[test]
    fn test_multiply_values3_round_trip() {
        let p = Polynomial::with_coefficients(vec![
            Scalar::from_const(1),
            Scalar::from_const(2),
            Scalar::from_const(3),
        ]);
        let q = Polynomial::with_coefficients(vec![
            Scalar::from_const(4),
            Scalar::from_const(5),
            Scalar::from_const(6),
        ]);
        let lhs = vec![
            p.evaluate_on_three_adic_domain(0, 3),
            p.evaluate_on_three_adic_domain(1, 3),
            p.evaluate_on_three_adic_domain(2, 3),
        ];
        let rhs = vec![
            q.evaluate_on_three_adic_domain(0, 3),
            q.evaluate_on_three_adic_domain(1, 3),
            q.evaluate_on_three_adic_domain(2, 3),
        ];
        let product = p.clone().multiply(q.clone());
        let result = Polynomial::encode3(Polynomial::multiply_values3(lhs, rhs));
        assert_eq!(result, product);
    }

    #[test]
    fn test_lagrange0_1() {
        let n = 1;
        let l0 = Polynomial::lagrange0(n);
        assert_eq!(l0.evaluate(Scalar::from_const(1)), Scalar::from_const(1));
    }

    #[test]
    fn test_lagrange0_2() {
        let n = 2;
        let omega = Polynomial::domain_element2(1, n);
        let l0 = Polynomial::lagrange0(n);
        assert_eq!(l0.evaluate(Scalar::from_const(1)), Scalar::from_const(1));
        assert_eq!(l0.evaluate(omega), Scalar::from_const(0));
    }

    #[test]
    fn test_lagrange0_4() {
        let n = 4;
        let omega = Polynomial::domain_element2(1, n);
        let l0 = Polynomial::lagrange0(n);
        assert_eq!(l0.evaluate(Scalar::from_const(1)), Scalar::from_const(1));
        assert_eq!(l0.evaluate(omega), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.square()), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.cube()), Scalar::from_const(0));
    }

    #[test]
    fn test_lagrange0_8() {
        let n = 8;
        let omega = Polynomial::domain_element2(1, n);
        let l0 = Polynomial::lagrange0(n);
        assert_eq!(l0.evaluate(Scalar::from_const(1)), Scalar::from_const(1));
        assert_eq!(l0.evaluate(omega), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(2)), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(3)), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(4)), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(5)), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(6)), Scalar::from_const(0));
        assert_eq!(l0.evaluate(omega.pow_small(7)), Scalar::from_const(0));
    }
}

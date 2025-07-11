use crate::prelude::*;
use std::collections::HashMap;

/// DIIS pop strategy.
pub enum DIISPopStrategy {
    /// Pop the vector with the largest iteration number.
    Iteration,

    /// Pop the vector with the largest diagonal element of the overlap matrix.
    ErrDiagonal,
}

pub struct DIISIncoreFlags {
    /// Maximum number of vectors in the DIIS space. Default is 6.
    pub space: usize,

    /// Minimum number of vectors in the DIIS space for extrapolation. Default is 2.
    pub min_space: usize,

    /// DIIS pop strategy. Default is `DIISPopStrategy::ErrDiagonal`.
    pub pop_strategy: DIISPopStrategy,
}

impl Default for DIISIncoreFlags {
    fn default() -> Self {
        Self { space: 6, min_space: 2, pop_strategy: DIISPopStrategy::ErrDiagonal }
    }
}

pub struct DIISIncoreIntermediates {
    /// The previous index of the inserted vector.
    prev: Option<usize>,

    /// Error vector overlap matrix for DIIS
    ///
    /// This overlap matrix follows convention that
    /// - the first row and column is auxiliary vector `[0, 1, ..., 1]`;
    /// - the rest of the matrix is the overlap matrix of the error vectors.
    ///
    /// Thus, internal index is 1-based.
    ovlp: Tsr,

    /// The zero-th vector inserted to the DIIS space.
    ///
    /// This is only used when error is not given, and the error is defined as the vector difference itself.
    /// Since the zero-th iteration does not have vector difference, we need to store the zero-th vector.
    vec_prev: Option<Tsr>,

    /// Error vectors for DIIS.
    ///
    /// Mapping: idx_internal -> err
    err_map: HashMap<usize, Tsr>,

    /// Vectors to be extrapolated for DIIS
    ///
    /// Mapping: idx_internal -> vec
    vec_map: HashMap<usize, Tsr>,

    /// Mapping of internal index.
    ///
    /// Mapping: idx_internal -> iteration
    niter_map: HashMap<usize, usize>,
}

pub struct DIISIncore {
    pub flags: DIISIncoreFlags,
    pub intermediates: DIISIncoreIntermediates,
}

/* #region logger */

/* #endregion */

#[allow(clippy::useless_conversion)]
impl DIISIncore {
    /// Initialize DIIS object.
    pub fn new(flags: DIISIncoreFlags, device: &DeviceTsr) -> Self {
        // initialize intermediates
        let mut ovlp = rt::zeros(([flags.space + 1, flags.space + 1], device));
        ovlp.i_mut((0, 1..)).fill(1.0);
        ovlp.i_mut((1.., 0)).fill(1.0);
        let intermediates = DIISIncoreIntermediates {
            prev: None,
            ovlp,
            vec_prev: None,
            err_map: HashMap::new(),
            vec_map: HashMap::new(),
            niter_map: HashMap::new(),
        };

        Self { flags, intermediates }
    }

    /// Compute the head by iteration number.
    pub fn get_head_by_iteration(&self) -> Option<usize> {
        let cur_space = self.intermediates.err_map.len();

        if cur_space == 0 {
            // No previously inserted vectors.
            None
        } else if cur_space < self.flags.space {
            // The head to be inserted is has not been filled.
            let idx_next = cur_space + 1;
            Some(idx_next)
        } else {
            // We assume that number of vectors should be larger than 1, so unwrap here.
            let key = self.intermediates.niter_map.iter().min_by(|a, b| a.1.cmp(b.1));
            let idx_next = *key.unwrap().0;
            Some(idx_next)
        }
    }

    /// Compute the head by diagonal of overlap.
    pub fn get_head_by_diagonal(&self) -> Option<usize> {
        let cur_space = self.intermediates.err_map.len();

        if cur_space == 0 {
            // No previously inserted vectors.
            None
        } else if cur_space < self.flags.space {
            // The head to be inserted is has not been filled.
            let idx_next = cur_space + 1;
            Some(idx_next)
        } else {
            // We assume that number of vectors should be larger than 1, so unwrap here.
            // Find the index of largest diagonal element of the overlap matrix.
            let ovlp = &self.intermediates.ovlp;
            let diagonal = ovlp.diagonal(None).abs();
            let idx_argmax = diagonal.argmax();
            if idx_argmax == 0 || idx_argmax > self.flags.space {
                // Error of vectors is too small, which is virtually impossible.
                // Evaluate next index by iteration number, but we will not raise error here.
                return self.get_head_by_iteration();
            }
            Some(idx_argmax)
        }
    }

    /// Compute the next index to be inserted.
    pub fn get_head(&self) -> Option<usize> {
        // get the head if not internally defined
        // if head is not defined, we will define it by the given strategy.
        match self.flags.pop_strategy {
            DIISPopStrategy::Iteration => self.get_head_by_iteration(),
            DIISPopStrategy::ErrDiagonal => self.get_head_by_diagonal(),
        }
    }

    /// Pop the head index and update the internal overlap of error vectors.
    ///
    /// - `None` will pop the internal evaluated index.
    /// - `Some(idx)` will pop the given index with the given stragety defined in flag `DIISFlags::pop_stragety`.
    ///
    /// Note that we do not assume the index is valid, so we will not raise error if the index is invalid.
    pub fn pop_head(&mut self, head: Option<usize>) {
        // Find the index to be popped.
        let head = head.or(self.get_head());

        if let Some(head) = head {
            // Actually pop the vector.
            self.intermediates.err_map.remove(&head);
            self.intermediates.vec_map.remove(&head);
            self.intermediates.niter_map.remove(&head);

            // Update the overlap matrix.
            let ovlp = &mut self.intermediates.ovlp;
            ovlp.i_mut((head, 1..)).fill(0.0);
            ovlp.i_mut((1.., head)).fill(0.0);
        }
    }

    /// Insert a vector to the DIIS space.
    pub fn insert(&mut self, vec: Tsr, head: Option<usize>, err: Option<Tsr>, iteration: Option<usize>) {
        // 1. unwrap head
        let head = head.or(self.get_head());
        let prev = self.intermediates.prev;

        // specical case: if head is the same to prev, then it means the last extrapolated vector has the maximum error;
        // then the function will infinite loops if remove the maximum error vector; so some code need to avoid this case.
        let head = if head == prev && head.is_some() {
            eprintln!(concat!(
                "DIIS error seems not good.\n",
                "The DIIS head is the same to the previous vector.\n",
                "It means that the last extrapolated vector has the maximum error.\n",
                "It is certainly not desired in DIIS, you may want to make a double-check.\n",
                "We will remove the vector with earliest iteration instead of the largest error norm."
            ));
            self.get_head_by_iteration()
        } else {
            head
        };

        // get index that will be inserted
        let head = head.unwrap_or(1);

        // pop head if necessary
        if self.intermediates.err_map.len() >= self.flags.space {
            self.pop_head(Some(head));
        }

        // 2. prepare error and vector
        let vec = vec.into_shape(-1);
        let err = match err {
            // a. if error is given, reshape it to 1D
            Some(err) => err.into_shape(-1),
            None => match &self.intermediates.vec_prev {
                Some(vec_prev) => &vec - vec_prev,
                None => {
                    self.intermediates.vec_prev = Some(vec);
                    return;
                },
            },
        };

        // 3. prepare iteration
        let iteration = iteration.unwrap_or({
            match prev {
                Some(prev) => self.intermediates.niter_map.get(&prev).unwrap() + 1,
                None => 0,
            }
        });

        // 4. insert the vector and update information
        self.intermediates.err_map.insert(head, err);
        self.intermediates.vec_map.insert(head, vec);
        self.intermediates.niter_map.insert(head, iteration);
        self.intermediates.prev = Some(head);

        // 5. update the overlap matrix
        let ovlp = &mut self.intermediates.ovlp;
        let err_cur = self.intermediates.err_map.get(&head).unwrap();
        let num_space = self.intermediates.err_map.len();
        let err_list =
            (1..=num_space).into_iter().map(|i| self.intermediates.err_map.get(&i).unwrap()).collect::<Vec<_>>();
        const CHUNK_SIZE: usize = 16384;
        let ovlp_cur = Self::incore_inner_dot(err_cur, &err_list, CHUNK_SIZE);
        ovlp.i_mut((head, 1..num_space + 1)).assign(&ovlp_cur);
        ovlp.i_mut((1..num_space + 1, head)).assign(&ovlp_cur.conj());
    }

    /// Extrapolate the vector from the DIIS space.
    pub fn extrapolate(&mut self) -> Tsr {
        // 1. get the number of vectors in the DIIS space
        let num_space = self.intermediates.err_map.len();
        if num_space == 0 {
            // no vectors in the DIIS space
            if self.intermediates.vec_prev.is_some() {
                return self.intermediates.vec_prev.as_ref().unwrap().to_owned();
            } else {
                // no vectors in the DIIS space and no zero-th vector
                // this is considered as error
                panic!("No vectors in the DIIS space. This may be an internal error.");
            }
        }

        // 1.5 not enough vectors for extrapolation
        if num_space < self.flags.min_space {
            let prev = self.intermediates.prev.unwrap();
            return self.intermediates.vec_map.get(&prev).unwrap().to_owned();
        }

        // 2. get the coefficients
        let ovlp = &self.intermediates.ovlp.i((..num_space + 1, ..num_space + 1));

        let (w, v) = rt::linalg::eigh(ovlp).into();

        let eps = 30.0 * f64::EPSILON;

        // set the small eigenvalues to inf, then take repciprocals
        let w = w.mapv(|x| if x.abs() < eps { 0.0 } else { 1.0 / x });

        // g: [1, 0, 0, ..., 0]
        let mut g: Tsr = rt::zeros(([num_space + 1], ovlp.device()));
        g[[0]] = 1.0;

        // DIIS coefficients
        let c = (v.view() * w) % v.t().conj() % g;

        // 3. extrapolate the vector
        let mut vec = self.intermediates.vec_map.get(&1).unwrap().zeros_like();
        for idx in 1..=num_space {
            let vec_idx = self.intermediates.vec_map.get(&idx).unwrap();
            vec += vec_idx * c[[idx]];
        }

        vec
    }

    /// Update the DIIS space.
    pub fn update(&mut self, vec: Tsr, err: Option<Tsr>, iteration: Option<usize>) -> Tsr {
        self.insert(vec, None, err, iteration);
        let vec = self.extrapolate();
        self.intermediates.vec_prev = Some(vec.to_owned());
        vec
    }

    /// Perform inner dot for obtaining overlap.
    ///
    /// This performs `a.conj() % b` by chunk.
    /// Note that `a.conj()` will allocate a new memory buffer, so this also costs some L3 cache bandwidth.
    pub fn incore_inner_dot(a: &Tsr, b_list: &[&Tsr], chunk: usize) -> Tsr {
        let size = a.size();
        let nlist = b_list.len();
        let mut result = rt::zeros(([nlist], a.device()));

        for i in (0..size).step_by(chunk) {
            let a = a.slice(i..i + chunk).conj();
            for (n, b) in b_list.iter().enumerate() {
                let b = b.slice(i..i + chunk);
                result[[n]] += (&a % b).to_scalar();
            }
        }
        result
    }
}

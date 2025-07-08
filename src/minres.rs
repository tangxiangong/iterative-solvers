//! Minimal Residual method (MINRES).
#![allow(dead_code)]

use nalgebra::DVector;
use crate::MatrixOp;

/// Minimal Residual method
pub struct MINRES<'mat, Mat: MatrixOp> {
    mat: &'mat Mat,
    solution: DVector<f64>,

    // Krylov basis vectors
    v_prev: DVector<f64>,
    v_curr: DVector<f64>,
    v_next: DVector<f64>,

    // W = R * inv(V) is computed in 3-term recurrence
    w_prev: DVector<f64>,
    w_curr: DVector<f64>,
    w_next: DVector<f64>,

    // Vector of size 4, holding the active column of the Hessenberg matrix
    h: [f64; 4],
    // rhs is just two active values of the right-hand side
    rhs: [f64; 2],

    // Some Givens rotations
    c_prev: f64,
    s_prev: f64,
    c_curr: f64,
    s_curr: f64,

    tol: f64,
    resnorm: f64,
    iteration: usize,
}

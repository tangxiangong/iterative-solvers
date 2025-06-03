use nalgebra::{DMatrix, DVector};

pub fn diag(data: &[f64], offset: i32) -> DMatrix<f64> {
    if data.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    match offset {
        0 => DMatrix::from_diagonal(&DVector::from_column_slice(data)),
        offset => {
            let offset_usize = offset.unsigned_abs() as usize;
            let n = data.len() + offset_usize;
            let mut mat = DMatrix::zeros(n, n);

            unsafe {
                if offset > 0 {
                    for (idx, &val) in data.iter().enumerate() {
                        *mat.get_unchecked_mut((idx, idx + offset_usize)) = val;
                    }
                } else {
                    for (idx, &val) in data.iter().enumerate() {
                        *mat.get_unchecked_mut((idx + offset_usize, idx)) = val;
                    }
                }
            }
            mat
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diag() {
        let data = vec![1.0, 2.0, 3.0];
        let mat = diag(&data, 0);
        println!("{}", mat);
        let mat = diag(&data, 1);
        println!("{}", mat);
        let mat = diag(&data, -1);
        println!("{}", mat);
        let mat = diag(&data, 2);
        println!("{}", mat);
        let mat = diag(&data, -2);
        println!("{}", mat);
    }
}

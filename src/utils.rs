/// Returns the smallest power of three that is >= n (returns 1 for n=0).
pub(crate) fn next_power_of_three(n: usize) -> usize {
    let mut pow = 1usize;
    while pow < n {
        pow *= 3;
    }
    pow
}

/// Checks if a number is a power of 3.
pub(crate) fn is_power_of_three(mut value: usize) -> bool {
    if value == 0 {
        return false;
    }
    while value > 1 {
        if value % 3 != 0 {
            return false;
        }
        value /= 3;
    }
    true
}

/// Computes the integer base-3 logarithm of `n`. For example, `ilog3(9) == 2`.
///
/// If `n` is not a power of 3 this function returns the logarithm rounded down to the nearest
/// integer, eg. `ilog3(8) == 1`.
pub(crate) fn ilog3(mut n: usize) -> usize {
    let mut c = 0;
    while n >= 3 {
        c += 1;
        n /= 3;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_power_of_three() {
        assert_eq!(next_power_of_three(0), 1);
        assert_eq!(next_power_of_three(1), 1);
        assert_eq!(next_power_of_three(2), 3);
        assert_eq!(next_power_of_three(3), 3);
        assert_eq!(next_power_of_three(4), 9);
        assert_eq!(next_power_of_three(5), 9);
        assert_eq!(next_power_of_three(6), 9);
        assert_eq!(next_power_of_three(7), 9);
        assert_eq!(next_power_of_three(8), 9);
        assert_eq!(next_power_of_three(9), 9);
        assert_eq!(next_power_of_three(10), 27);
        assert_eq!(next_power_of_three(11), 27);
    }

    #[test]
    fn test_is_power_of_three() {
        assert!(!is_power_of_three(0));
        assert!(is_power_of_three(1));
        assert!(!is_power_of_three(2));
        assert!(is_power_of_three(3));
        assert!(!is_power_of_three(4));
        assert!(!is_power_of_three(5));
        assert!(!is_power_of_three(6));
        assert!(!is_power_of_three(7));
        assert!(!is_power_of_three(8));
        assert!(is_power_of_three(9));
        assert!(!is_power_of_three(10));
        assert!(!is_power_of_three(11));
    }

    #[test]
    fn test_ilog3() {
        assert_eq!(ilog3(0), 0);
        assert_eq!(ilog3(1), 0);
        assert_eq!(ilog3(2), 0);
        assert_eq!(ilog3(3), 1);
        assert_eq!(ilog3(4), 1);
        assert_eq!(ilog3(5), 1);
        assert_eq!(ilog3(6), 1);
        assert_eq!(ilog3(7), 1);
        assert_eq!(ilog3(8), 1);
        assert_eq!(ilog3(9), 2);
        assert_eq!(ilog3(10), 2);
        assert_eq!(ilog3(11), 2);
    }
}

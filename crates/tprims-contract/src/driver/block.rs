//! Axis-aware outer blocks: the membership the blocked traversal needs.
//!
//! Today the driver walks consecutive slices of one flattened row scatter
//! (`static_grid.rs`, `tile.rs`). While the panel's row order is the output's that
//! is all it needs. It is not enough when the packed operand and the output
//! disagree about their fastest axis: a consecutive slice of the source-ordered
//! flattening then holds the output's fast axis only once per
//! `product_of_the_other_axes` rows, so no reordering *inside* that slice can
//! produce a contiguous destination run. The campaign's `abjc-cbka-kj` is such a
//! contraction, and the design in `docs/design/blocked-outer-traversal.md` needs a
//! block that spans whole runs of *both* axes.
//!
//! This module is that membership, plus the conservative predicate that keeps
//! every other shape on today's path. Nothing calls it yet: it is the first,
//! behaviour-preserving step of that design, and the tests below pin both the
//! identity view of today's slices and the runs the measured case needs.

use crate::plan::Axis;

/// How many values a block takes along an axis: the output's fastest axis whole,
/// the operand's fastest axis up to one cache line, everything else one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) struct BlockShape {
    /// Index into the role's axis list of the axis a block spans whole.
    pub(crate) span: usize,
    /// Index of the axis a block takes a line's worth of.
    pub(crate) line: usize,
    /// Values taken along `line`'s axis, `1..=extent`.
    pub(crate) line_len: usize,
}

/// The role's axes ordered by an operand's stride, fastest first. `None` when the
/// role has fewer than two axes, or when an axis is singleton (nothing to gain
/// and no order to preserve).
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
fn fastest(axes: &[Axis], stride: impl Fn(&Axis) -> i64) -> Option<usize> {
    let live: Vec<usize> = (0..axes.len()).filter(|&i| axes[i].extent > 1).collect();
    if live.len() < 2 {
        return None;
    }
    live.into_iter()
        .min_by_key(|&i| stride(&axes[i]).unsigned_abs())
}

/// The block a role needs, or `None` when today's slices are the right answer.
///
/// `operand`, `output`: the two stride projections of the role's axes.
/// `per_line`: elements a 64-byte line holds at this element size (8 for f64), so
/// that a block takes exactly one line's worth along the operand's fastest axis.
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) fn eligible(
    axes: &[Axis],
    operand: impl Fn(&Axis) -> i64,
    output: impl Fn(&Axis) -> i64,
    per_line: usize,
) -> Option<BlockShape> {
    let fast_op = fastest(axes, &operand)?;
    let fast_out = fastest(axes, &output)?;
    if fast_op == fast_out {
        // The same axis is fastest for both: today's slices already keep the
        // operand's runs together, so there is nothing to change.
        return None;
    }
    Some(BlockShape {
        span: fast_out,
        line: fast_op,
        line_len: (axes[fast_op].extent as usize).min(per_line.max(1)),
    })
}

/// The block's global flattened logical row indices, in the operand's order: the
/// axis a block takes a line's worth of varies fastest, then the rest of the
/// block's axes in the role's own list order.
///
/// The indices are mixed-radix over the role's axes, which is the numbering the
/// plan's scatters use, so they can index every operand's scatter - the element
/// offsets differ per operand, the logical row does not.
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) fn rows(
    axes: &[Axis],
    shape: BlockShape,
    span_start: usize,
    line_start: usize,
) -> Vec<i64> {
    let mut strides = vec![0i64; axes.len()];
    let mut acc = 1i64;
    for (i, a) in axes.iter().enumerate() {
        strides[i] = acc;
        acc *= a.extent;
    }
    let n = shape.line_len * axes[shape.span].extent as usize;
    let mut out = Vec::with_capacity(n);
    for s in 0..axes[shape.span].extent as usize {
        for l in 0..shape.line_len {
            let mut row = 0i64;
            for (i, &st) in strides.iter().enumerate() {
                let v = if i == shape.line {
                    line_start + l
                } else if i == shape.span {
                    span_start + s
                } else {
                    0
                };
                row += v as i64 * st;
            }
            out.push(row);
        }
    }
    out
}

/// Gather one operand's element offsets for a block's rows.
#[allow(dead_code)] // the enumerator that consumes it lands with the design's next step
pub(crate) fn gather(scatter: &[i64], rows: &[i64]) -> Vec<i64> {
    rows.iter().map(|&r| scatter[r as usize]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured case: `A` is `(a s1, b s48, j s1920, c s92160)` and the output
    /// is `(c s1, b s48, n s1920, a s76800)`, so the m role's axes are `(a, b, c)`
    /// with those strides.
    fn measured() -> Vec<Axis> {
        vec![
            Axis {
                extent: 48,
                sa: 1,
                sb: 0,
                sc: 1,
                sd: 76800,
            },
            Axis {
                extent: 40,
                sa: 48,
                sb: 0,
                sc: 48,
                sd: 48,
            },
            Axis {
                extent: 48,
                sa: 92160,
                sb: 0,
                sc: 76800,
                sd: 1,
            },
        ]
    }

    /// The output offsets of the same case for a block's rows, decoded from the
    /// plan's mixed-radix numbering (a + 48 b + 1920 c).
    fn out_offsets(rows: &[i64]) -> Vec<i64> {
        rows.iter()
            .map(|&r| (r % 48) * 76800 + (r / 48) % 40 * 48 + (r / 1920))
            .collect()
    }

    #[test]
    fn the_measured_case_takes_a_line_of_a_and_a_whole_c() {
        let axes = measured();
        let shape = eligible(&axes, |x| x.sa, |x| x.sd, 8).expect("eligible");
        assert_eq!(shape.span, 2, "the output's fastest axis is c");
        assert_eq!(shape.line, 0, "the operand's fastest axis is a");
        assert_eq!(shape.line_len, 8, "one 64 B line of f64");
        let rows = rows(&axes, shape, 0, 0);
        assert_eq!(rows.len(), 8 * 48);
        // Consecutive rows are consecutive a: the source run the pack reads.
        assert_eq!(rows[1] - rows[0], 1);
        // And every a comes with all 48 c's, whose output offsets are consecutive:
        // the destination run the emitter writes. This is the property the
        // flattened slice could not provide and the whole design rests on.
        let offs = out_offsets(&rows);
        for a in 0..8i64 {
            let mut run: Vec<i64> = rows
                .iter()
                .zip(&offs)
                .filter(|(r, _)| **r % 1920 == a)
                .map(|(_, o)| *o)
                .collect();
            run.sort_unstable();
            assert_eq!(run.len(), 48, "each a carries all 48 c's");
            assert_eq!(
                run.last().unwrap() - run.first().unwrap(),
                47,
                "consecutive"
            );
        }
    }

    #[test]
    fn a_role_whose_fastest_axis_agrees_stays_on_todays_path() {
        let axes = measured();
        assert!(
            eligible(&axes, |x| x.sa, |x| x.sa, 8).is_none(),
            "same fastest axis: nothing to change"
        );
        let single = vec![Axis {
            extent: 92160,
            sa: 1,
            sb: 0,
            sc: 1,
            sd: 76800,
        }];
        assert!(
            eligible(&single, |x| x.sa, |x| x.sd, 8).is_none(),
            "one axis"
        );
    }

    #[test]
    fn gather_is_the_identity_for_todays_intervals() {
        let scatter: Vec<i64> = (0..24).map(|i| i * 7).collect();
        let rows: Vec<i64> = (0..24).collect();
        assert_eq!(gather(&scatter, &rows), scatter);
    }
}

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

use crate::plan::PlanStats;

/// One axis of an oriented role as the blocked path needs it: its extent and its
/// stride in the packed operand and in the output. `sc`/`sd` for the m role,
/// `sb`/`sd` for the n role, and the roles swap under orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // as below: the enumerator that consumes it lands next
pub(crate) struct Role {
    pub(crate) extent: usize,
    pub(crate) src: i64,
    pub(crate) dst: i64,
}

/// The oriented role's axes, in the plan's own order. `swapped`: the plan
/// exchanges the row and column operands, so the row role is the user's `n`.
#[allow(dead_code)]
pub(crate) fn role(stats: &PlanStats, swapped: bool) -> Vec<Role> {
    let axes = if swapped {
        &stats.n_axes
    } else {
        &stats.m_axes
    };
    axes.iter()
        .map(|a| Role {
            extent: a.extent as usize,
            src: if swapped { a.sb } else { a.sa },
            dst: a.sd,
        })
        .collect()
}

/// Whether the blocked path applies at all: the oriented row role's fastest axes
/// disagree (so today's consecutive slices cannot hold both runs), the contraction
/// fits **one K slab** and **one NC panel** (so a block's accumulator is complete
/// without crossing slabs and the block's whole output is one contiguous write),
/// and the role has at least two live axes. Everything else keeps today's path.
///
/// The two slab conditions are what keep this the narrow, safe half of the design:
/// no cross-slab persistence, no ownership or barrier change, no resized panels.
#[allow(dead_code)]
pub(crate) fn blocked_eligibility(
    stats: &PlanStats,
    swapped: bool,
    k: usize,
    n: usize,
    kc: usize,
    nc: usize,
    per_line: usize,
) -> Option<BlockShape> {
    if k > kc || n > nc {
        return None;
    }
    eligible(&role(stats, swapped), per_line)
}

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
fn fastest(axes: &[Role], stride: impl Fn(&Role) -> i64) -> Option<usize> {
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
pub(crate) fn eligible(axes: &[Role], per_line: usize) -> Option<BlockShape> {
    let (operand, output) = (|a: &Role| a.src, |a: &Role| a.dst);
    let fast_op = fastest(axes, operand)?;
    let fast_out = fastest(axes, output)?;
    if fast_op == fast_out {
        // The same axis is fastest for both: today's slices already keep the
        // operand's runs together, so there is nothing to change.
        return None;
    }
    Some(BlockShape {
        span: fast_out,
        line: fast_op,
        line_len: axes[fast_op].extent.min(per_line.max(1)),
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
    axes: &[Role],
    shape: BlockShape,
    span_start: usize,
    line_start: usize,
) -> Vec<i64> {
    let mut strides = vec![0i64; axes.len()];
    let mut acc = 1i64;
    for (i, a) in axes.iter().enumerate() {
        strides[i] = acc;
        acc *= a.extent as i64;
    }
    let n = shape.line_len * axes[shape.span].extent;
    let mut out = Vec::with_capacity(n);
    for s in 0..axes[shape.span].extent {
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

/// The order in which a block's rows must be visited for the output stores to be
/// consecutive: `d_local`'s row indices sorted by their element offset, stably, so
/// that rows sharing an offset keep the block's own order (the raw write-back
/// contract's last-writer behaviour for repeated addresses depends on it).
///
/// This is the other half of the design: the block's membership supplies the
/// source run the pack reads, and this order turns the block's accumulated tile
/// into the sequence the emitter writes, which is why the transposition can be
/// paid once per block instead of once per micro-tile.
#[allow(dead_code)] // as above: the enumerator that consumes it lands next
pub(crate) fn output_order(d_local: &[i64]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..d_local.len() as u32).collect();
    order.sort_by_key(|&r| d_local[r as usize]);
    order
}

/// Reorder a block's accumulated rows into `output_order`: `src` and `dst` are
/// row-major `rows x nc`, and `src`'s row `r` becomes `dst`'s row `order[r]`'s
/// source - i.e. `dst[i] = src[order[i]]` row by row. Written as a plain copy so
/// it can be a model for the driver's tiled version.
#[allow(dead_code)] // as above: the enumerator that consumes it lands next
pub(crate) fn reorder_rows<T: Copy>(src: &[T], nc: usize, order: &[u32], dst: &mut [T]) {
    assert_eq!(src.len(), order.len() * nc, "block scratch shape");
    assert_eq!(dst.len(), src.len(), "same block, reordered");
    for (i, &r) in order.iter().enumerate() {
        let (s, d) = (r as usize * nc, i * nc);
        dst[d..d + nc].copy_from_slice(&src[s..s + nc]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Axis;

    /// The measured case's m role: `A` is `(a s1, b s48, j s1920, c s92160)` and the
    /// output is `(c s1, b s48, n s1920, a s76800)`.
    fn measured() -> Vec<Role> {
        vec![
            Role {
                extent: 48,
                src: 1,
                dst: 76800,
            },
            Role {
                extent: 40,
                src: 48,
                dst: 48,
            },
            Role {
                extent: 48,
                src: 92160,
                dst: 1,
            },
        ]
    }

    /// The same case as the plan reports it, for the eligibility test.
    fn stats() -> crate::plan::PlanStats {
        let ax = |extent: usize, sa: i64, sc: i64, sd: i64| Axis {
            extent: extent as i64,
            sa,
            sb: 0,
            sc,
            sd,
        };
        crate::plan::PlanStats {
            m: 92160,
            n: 40,
            k: 48,
            batch: 1,
            m_axes: vec![
                ax(48, 1, 1, 76800),
                ax(40, 48, 48, 48),
                ax(48, 92160, 76800, 1),
            ],
            n_axes: vec![ax(40, 0, 1920, 1920)],
            k_axes: vec![ax(48, 1920, 0, 0)],
            h_axes: vec![],
            is_pure_gemm: false,
        }
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
        let shape = eligible(&axes, 8).expect("eligible");
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
        let same = vec![
            Role {
                extent: 48,
                src: 1,
                dst: 1,
            },
            Role {
                extent: 48,
                src: 92160,
                dst: 76800,
            },
        ];
        assert!(eligible(&same, 8).is_none(), "same fastest axis");
        assert!(eligible(&axes[..1], 8).is_none(), "one live axis");
    }

    #[test]
    fn eligibility_needs_one_k_slab_and_one_nc_panel() {
        let s = stats();
        // The measured case: k = 48 <= kc = 256 and n = 40 <= nc = 1536.
        assert!(blocked_eligibility(&s, false, 48, 40, 256, 1536, 8).is_some());
        assert!(
            blocked_eligibility(&s, false, 48, 40, 32, 1536, 8).is_none(),
            "two K slabs"
        );
        assert!(
            blocked_eligibility(&s, false, 48, 40, 256, 16, 8).is_none(),
            "two NC panels"
        );
        // The n role is a single axis, so a swapped plan has nothing to block.
        assert!(blocked_eligibility(&s, true, 48, 40, 256, 1536, 8).is_none());
    }

    #[test]
    fn gather_is_the_identity_for_todays_intervals() {
        let scatter: Vec<i64> = (0..24).map(|i| i * 7).collect();
        let rows: Vec<i64> = (0..24).collect();
        assert_eq!(gather(&scatter, &rows), scatter);
    }

    #[test]
    fn the_output_order_is_consecutive_for_the_measured_case() {
        let axes = measured();
        let shape = eligible(&axes, 8).expect("eligible");
        let rows = rows(&axes, shape, 0, 0);
        let offs = out_offsets(&rows);
        // The emitter is handed the block's rows in this order, so its stores run
        // consecutively in runs of 48 - one per `a`, whose `c` values are the
        // output's contiguous axis.
        let order = output_order(&offs);
        let ordered: Vec<i64> = order.iter().map(|&r| offs[r as usize]).collect();
        assert!(
            ordered.windows(2).all(|w| w[0] <= w[1]),
            "sorted by output offset"
        );
        let runs: Vec<usize> = {
            let mut runs = Vec::new();
            let mut n = 1usize;
            for w in ordered.windows(2) {
                if w[1] == w[0] + 1 {
                    n += 1;
                } else {
                    runs.push(n);
                    n = 1;
                }
            }
            runs.push(n);
            runs
        };
        assert_eq!(runs.len(), 8, "one contiguous run per a");
        assert!(
            runs.iter().all(|&n| n >= 48),
            "each run is the block's 48 c's"
        );
    }

    #[test]
    fn the_output_order_is_the_identity_when_the_block_is_already_ordered() {
        let offs: Vec<i64> = (0..16).map(|i| i * 3).collect();
        assert_eq!(output_order(&offs), (0..16).collect::<Vec<u32>>());
    }

    #[test]
    fn ties_keep_the_blocks_own_order() {
        // The raw write-back contract allows repeated target addresses; the order
        // must not silently reorder them.
        let offs = vec![5i64, 1, 5, 1, 5];
        assert_eq!(output_order(&offs), vec![1u32, 3, 0, 2, 4]);
    }

    #[test]
    fn reorder_rows_moves_whole_rows() {
        let src: Vec<i64> = (0..6).collect(); // 3 rows x 2 cols
        let mut dst = vec![0i64; 6];
        reorder_rows(&src, 2, &[2, 0, 1], &mut dst);
        assert_eq!(dst, vec![4, 5, 0, 1, 2, 3]);
    }
}

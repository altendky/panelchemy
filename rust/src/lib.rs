//! Exact continuous bidirectional source-distance certification.
//!
//! Floating-point boxes select possible witnesses only. Every acceptance
//! predicate uses the exact rational values of the original binary floats.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rstar::{AABB, RTree, RTreeObject};

type Rational = BigRational;
type Point = [Rational; 2];
type Segment = (Point, Point);
type Controls = [Point; 4];
type FloatPoint = [f64; 2];
type FloatSpan = ([FloatPoint; 4], bool);

const INVALID_INPUT: &str =
    "distance certification requires finite planar paths and nonnegative tolerance";

fn integer(value: i64) -> Rational {
    Rational::from_integer(BigInt::from(value))
}

fn point(value: FloatPoint) -> Point {
    value.map(|v| Rational::from_float(v).expect("finite input was validated"))
}

fn difference(a: &Point, b: &Point) -> Point {
    [&a[0] - &b[0], &a[1] - &b[1]]
}

fn squared(point: &Point) -> Rational {
    &point[0] * &point[0] + &point[1] * &point[1]
}

fn dot(a: &Point, b: &Point) -> Rational {
    &a[0] * &b[0] + &a[1] * &b[1]
}

fn cross(a: &Point, b: &Point) -> Rational {
    &a[0] * &b[1] - &a[1] * &b[0]
}

fn mix(a: &Point, b: &Point, t: &Rational) -> Point {
    [&a[0] + (&b[0] - &a[0]) * t, &a[1] + (&b[1] - &a[1]) * t]
}

fn split(controls: &Controls) -> (Controls, Controls) {
    let half = Rational::new(BigInt::from(1), BigInt::from(2));
    let ab = mix(&controls[0], &controls[1], &half);
    let bc = mix(&controls[1], &controls[2], &half);
    let cd = mix(&controls[2], &controls[3], &half);
    let abc = mix(&ab, &bc, &half);
    let bcd = mix(&bc, &cd, &half);
    let middle = mix(&abc, &bcd, &half);
    (
        [controls[0].clone(), ab, abc, middle.clone()],
        [middle, bcd, cd, controls[3].clone()],
    )
}

fn controls(value: FloatSpan) -> Controls {
    let (values, is_line) = value;
    let mut points = values.map(point);
    if is_line {
        let third = Rational::new(BigInt::from(1), BigInt::from(3));
        points[1] = mix(&points[0], &points[3], &third);
        points[2] = mix(&points[0], &points[3], &(integer(2) * third));
    }
    points
}

fn inside(point: &Point, segment: &Segment, radius_squared: &Rational) -> bool {
    let (a, b) = segment;
    let direction = difference(b, a);
    let offset = difference(point, a);
    let length = squared(&direction);
    let projection = dot(&offset, &direction);
    if projection <= Rational::zero() {
        return squared(&offset) <= *radius_squared;
    }
    if projection >= length {
        return squared(&difference(point, b)) <= *radius_squared;
    }
    let determinant = cross(&offset, &direction);
    &determinant * &determinant <= radius_squared * length
}

fn straight(controls: &Controls) -> bool {
    let a = &controls[0];
    let direction = difference(&controls[3], a);
    let length = squared(&direction);
    if length.is_zero() {
        return controls.iter().all(|point| point == a);
    }
    controls[1..3].iter().all(|point| {
        let offset = difference(point, a);
        let projection = dot(&offset, &direction);
        cross(&offset, &direction).is_zero()
            && projection >= Rational::zero()
            && projection <= length
    })
}

#[derive(Clone)]
struct IndexedBounds {
    envelope: AABB<[f64; 2]>,
    index: usize,
}

impl RTreeObject for IndexedBounds {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.envelope
    }
}

fn float(value: &Rational) -> f64 {
    value.to_f64().unwrap_or(if value.is_negative() {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    })
}

struct Capsules {
    segments: Vec<Segment>,
    squared: Vec<Rational>,
    radius: Rational,
    bounds: Vec<[f64; 4]>,
    index: Option<RTree<IndexedBounds>>,
}

impl Capsules {
    fn new(segments: Vec<Segment>, radii: Vec<Rational>, index_min_segments: usize) -> Self {
        let radius = radii.iter().max().expect("nonempty capsules").clone();
        let squared = radii.iter().map(|r| r * r).collect();
        let bounds: Vec<[f64; 4]> = segments
            .iter()
            .map(|(a, b)| {
                [
                    float(std::cmp::min(&a[0], &b[0])).next_down(),
                    float(std::cmp::min(&a[1], &b[1])).next_down(),
                    float(std::cmp::max(&a[0], &b[0])).next_up(),
                    float(std::cmp::max(&a[1], &b[1])).next_up(),
                ]
            })
            .collect();
        // R-tree envelope arithmetic may overflow even for finite bounds.
        // Fall back to the same ordered linear broad phase in that case.
        let index = (segments.len() >= index_min_segments
            && bounds
                .iter()
                .flatten()
                .all(|value| value.is_finite() && value.abs() <= 1.0e150))
        .then(|| {
            RTree::bulk_load(
                bounds
                    .iter()
                    .enumerate()
                    .map(|(index, b)| IndexedBounds {
                        envelope: AABB::from_corners([b[0], b[1]], [b[2], b[3]]),
                        index,
                    })
                    .collect(),
            )
        });
        Self {
            segments,
            squared,
            radius,
            bounds,
            index,
        }
    }

    fn candidates(&self, points: &[Point]) -> Vec<usize> {
        let radius = float(&self.radius).next_up();
        let lower: [f64; 2] = std::array::from_fn(|axis| {
            let value = points.iter().map(|p| &p[axis]).min().expect("points");
            (float(value).next_down() - radius).next_down()
        });
        let upper: [f64; 2] = std::array::from_fn(|axis| {
            let value = points.iter().map(|p| &p[axis]).max().expect("points");
            (float(value).next_up() + radius).next_up()
        });
        // An overflow or NaN in float-only broad-phase arithmetic must never
        // hide an exact witness. The exact predicates remain authoritative.
        if lower.iter().chain(&upper).any(|value| value.is_nan()) {
            return (0..self.bounds.len()).collect();
        }
        let mut indices: Vec<usize> = match &self.index {
            Some(index) if lower.iter().chain(&upper).all(|value| value.is_finite()) => {
                let envelope = AABB::from_corners(lower, upper);
                index
                    .locate_in_envelope_intersecting(envelope)
                    .map(|entry| entry.index)
                    .collect()
            }
            _ => (0..self.bounds.len()).collect(),
        };
        indices.sort_unstable();
        indices.retain(|&i| {
            let [left, bottom, right, top] = self.bounds[i];
            right >= lower[0] && left <= upper[0] && top >= lower[1] && bottom <= upper[1]
        });
        indices
    }

    fn inside(&self, point: &Point, index: usize) -> bool {
        inside(point, &self.segments[index], &self.squared[index])
    }

    fn covered(&self, points: &[Point], candidates: &[usize]) -> bool {
        candidates
            .iter()
            .any(|&i| points.iter().all(|point| self.inside(point, i)))
    }

    fn parallel_cover(&self, a: &Point, b: &Point, candidates: &[usize]) -> bool {
        let direction = difference(b, a);
        let length = squared(&direction);
        if length.is_zero() {
            return self.covered(std::slice::from_ref(a), candidates);
        }
        let mut intervals = Vec::new();
        for &index in candidates {
            let (p, q) = &self.segments[index];
            if !cross(&direction, &difference(q, p)).is_zero() {
                continue;
            }
            let determinant = cross(&difference(p, a), &direction);
            if &determinant * &determinant > &self.squared[index] * &length {
                continue;
            }
            let u = dot(&difference(p, a), &direction) / &length;
            let v = dot(&difference(q, a), &direction) / &length;
            intervals.push((
                Rational::zero().max(u.clone().min(v.clone())),
                integer(1).min(u.max(v)),
            ));
        }
        intervals.sort();
        let mut reached = Rational::zero();
        for (start, stop) in intervals {
            if start > reached {
                return false;
            }
            reached = reached.max(stop);
            if reached >= integer(1) {
                return true;
            }
        }
        false
    }
}

struct Budget {
    remaining: usize,
    exhausted: bool,
}

impl Budget {
    fn spend(&mut self) -> bool {
        if self.remaining == 0 {
            self.exhausted = true;
            false
        } else {
            self.remaining -= 1;
            true
        }
    }
}

#[derive(Clone, Copy)]
struct Limits {
    max_nodes: usize,
    max_depth: usize,
    proxy_refinements: usize,
    index_min_segments: usize,
}

fn curve_to_source(
    controls: &Controls,
    target: &Capsules,
    budget: &mut Budget,
    max_depth: usize,
) -> bool {
    let mut pending = vec![(controls.clone(), 0)];
    while let Some((controls, depth)) = pending.pop() {
        if !budget.spend() {
            return false;
        }
        let candidates = target.candidates(&controls);
        if target.covered(&controls, &candidates)
            || (straight(&controls)
                && target.parallel_cover(&controls[0], &controls[3], &candidates))
        {
            continue;
        }
        if depth == max_depth
            || [0, 3]
                .iter()
                .any(|&j| !candidates.iter().any(|&i| target.inside(&controls[j], i)))
        {
            return false;
        }
        let (left, right) = split(&controls);
        pending.push((right, depth + 1));
        pending.push((left, depth + 1));
    }
    true
}

fn proxy(
    controls: &Controls,
    epsilon: &Rational,
    budget: &mut Budget,
    max_depth: usize,
) -> Option<Vec<Segment>> {
    let mut result = Vec::new();
    let epsilon_squared = epsilon * epsilon;
    let mut pending = vec![(controls.clone(), 0)];
    let third = Rational::new(BigInt::from(1), BigInt::from(3));
    while let Some((controls, depth)) = pending.pop() {
        if !budget.spend() {
            return None;
        }
        let a = &controls[0];
        let d = &controls[3];
        let chord = [
            a.clone(),
            mix(a, d, &third),
            mix(a, d, &(integer(2) * &third)),
            d.clone(),
        ];
        if controls
            .iter()
            .zip(&chord)
            .all(|(p, q)| squared(&difference(p, q)) <= epsilon_squared)
        {
            result.push((a.clone(), d.clone()));
        } else if depth == max_depth {
            return None;
        } else {
            let (left, right) = split(&controls);
            pending.push((right, depth + 1));
            pending.push((left, depth + 1));
        }
    }
    Some(result)
}

fn source_to_proxy(
    segment: &Segment,
    target: &Capsules,
    budget: &mut Budget,
    max_depth: usize,
) -> bool {
    let mut pending = vec![(segment.clone(), 0)];
    let half = Rational::new(BigInt::from(1), BigInt::from(2));
    while let Some(((a, b), depth)) = pending.pop() {
        if !budget.spend() {
            return false;
        }
        let points = [a, b];
        let candidates = target.candidates(&points);
        if target.covered(&points, &candidates)
            || target.parallel_cover(&points[0], &points[1], &candidates)
        {
            continue;
        }
        if depth == max_depth
            || points
                .iter()
                .any(|p| !candidates.iter().any(|&i| target.inside(p, i)))
        {
            return false;
        }
        let [a, b] = points;
        let middle = mix(&a, &b, &half);
        pending.push(((middle.clone(), b), depth + 1));
        pending.push(((a, middle), depth + 1));
    }
    true
}

fn certify(
    source_paths: Vec<Vec<FloatPoint>>,
    spans: Vec<FloatSpan>,
    tolerance: f64,
    limits: Limits,
) -> bool {
    let segments: Vec<Segment> = source_paths
        .into_iter()
        .flat_map(|path| {
            let points: Vec<Point> = path.into_iter().map(point).collect();
            points
                .windows(2)
                .map(|pair| (pair[0].clone(), pair[1].clone()))
                .collect::<Vec<_>>()
        })
        .collect();
    let tolerance = Rational::from_float(tolerance).expect("finite input validated");
    let mut target = Capsules::new(
        segments.clone(),
        vec![tolerance.clone(); segments.len()],
        limits.index_min_segments,
    );
    let mut budget = Budget {
        remaining: limits.max_nodes,
        exhausted: false,
    };
    let controls: Vec<Controls> = spans.into_iter().map(controls).collect();
    if !controls
        .iter()
        .all(|span| curve_to_source(span, &target, &mut budget, limits.max_depth))
    {
        return false;
    }
    let mut first_unproved = 0;
    let nonlinear = controls.iter().any(|span| !straight(span));
    let mut epsilon = &tolerance / integer(16);
    for _ in 0..=limits.proxy_refinements {
        let mut approximation = Vec::new();
        let mut radii = Vec::new();
        for span in &controls {
            if straight(span) {
                approximation.push((span[0].clone(), span[3].clone()));
                radii.push(tolerance.clone());
            } else {
                if tolerance.is_zero() {
                    return false;
                }
                let Some(selected) = proxy(span, &epsilon, &mut budget, limits.max_depth) else {
                    return false;
                };
                radii.extend(vec![&tolerance - &epsilon; selected.len()]);
                approximation.extend(selected);
            }
        }
        target = Capsules::new(
            approximation.clone(),
            radii.clone(),
            limits.index_min_segments,
        );
        while first_unproved < segments.len()
            && source_to_proxy(
                &segments[first_unproved],
                &target,
                &mut budget,
                limits.max_depth,
            )
        {
            first_unproved += 1;
        }
        if first_unproved == segments.len() {
            return true;
        }
        if budget.exhausted || !nonlinear {
            return false;
        }
        let outer = Capsules::new(
            approximation,
            radii.iter().map(|r| integer(2) * &tolerance - r).collect(),
            limits.index_min_segments,
        );
        let (a, b) = &segments[first_unproved];
        if [a, b].iter().any(|point| {
            let points = std::slice::from_ref(*point);
            !outer.covered(points, &outer.candidates(points))
        }) {
            return false;
        }
        epsilon /= integer(4);
    }
    false
}

/// The caller supplies Python's resource settings explicitly, keeping the
/// extension and fallback's proof budgets synchronized.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn within_paths(
    py: Python<'_>,
    source_paths: Vec<Vec<FloatPoint>>,
    spans: Vec<FloatSpan>,
    tolerance: f64,
    max_nodes: usize,
    max_depth: usize,
    proxy_refinements: usize,
    index_min_segments: usize,
) -> PyResult<bool> {
    if !tolerance.is_finite()
        || tolerance < 0.0
        || source_paths.is_empty()
        || source_paths.iter().any(|path| path.len() < 2)
        || spans.is_empty()
        || source_paths
            .iter()
            .flatten()
            .flatten()
            .any(|v| !v.is_finite())
        || spans
            .iter()
            .flat_map(|s| s.0)
            .flatten()
            .any(|v| !v.is_finite())
    {
        return Err(PyValueError::new_err(INVALID_INPUT));
    }
    let limits = Limits {
        max_nodes,
        max_depth,
        proxy_refinements,
        index_min_segments,
    };
    Ok(py.detach(move || certify(source_paths, spans, tolerance, limits)))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(within_paths, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> Limits {
        Limits {
            max_nodes: 100_000,
            max_depth: 24,
            proxy_refinements: 3,
            index_min_segments: 32,
        }
    }

    fn line(a: FloatPoint, b: FloatPoint) -> FloatSpan {
        // Finite unused handles are deliberately meaningless: is_line owns
        // exact rational degree elevation after input validation.
        ([a, [987.0, -321.0], [-123.0, 876.0], b], true)
    }

    #[test]
    fn preserves_exact_binary_float_values() {
        let tenth = point([0.1, -0.0]);
        assert_eq!(
            tenth[0],
            Rational::new(
                BigInt::from(3_602_879_701_896_397_i64),
                BigInt::from(36_028_797_018_963_968_i64)
            )
        );
        assert!(tenth[1].is_zero());
        let smallest = point([f64::from_bits(1), 0.0]);
        assert_eq!(smallest[0].denom(), &(BigInt::from(1) << 1074));
    }

    #[test]
    fn line_canonicalization_uses_exact_thirds() {
        let actual = controls(line([0.0, 0.0], [1.0, 0.0]));
        assert_eq!(
            actual[1][0],
            Rational::new(BigInt::from(1), BigInt::from(3))
        );
        assert_eq!(
            actual[2][0],
            Rational::new(BigInt::from(2), BigInt::from(3))
        );
    }

    #[test]
    fn zero_tolerance_covers_nondyadic_parallel_partitions() {
        let path = vec![vec![[0.0, 0.0], [0.3, 0.0], [0.7, 0.0], [1.0, 0.0]]];
        assert!(certify(
            path,
            vec![line([0.0, 0.0], [1.0, 0.0])],
            0.0,
            limits()
        ));
        assert!(certify(
            vec![vec![[0.0, 0.0], [1.0, 0.0]]],
            vec![line([0.0, 0.0], [0.3, 0.0]), line([0.3, 0.0], [1.0, 0.0])],
            0.0,
            limits()
        ));
    }

    #[test]
    fn disconnected_source_paths_do_not_gain_a_bridge() {
        let paths = vec![vec![[0.0, 0.0], [1.0, 0.0]], vec![[3.0, 0.0], [4.0, 0.0]]];
        assert!(certify(
            paths.clone(),
            vec![line([0.0, 0.0], [1.0, 0.0]), line([3.0, 0.0], [4.0, 0.0])],
            0.0,
            limits()
        ));
        assert!(!certify(
            paths,
            vec![line([0.0, 0.0], [4.0, 0.0])],
            0.0,
            limits()
        ));
    }

    #[test]
    fn directed_containment_alone_is_insufficient() {
        assert!(!certify(
            vec![vec![[0.0, 0.0], [2.0, 0.0]]],
            vec![line([0.0, 0.0], [1.0, 0.0])],
            0.1,
            limits()
        ));
        assert!(!certify(
            vec![vec![[0.0, 0.0], [1.0, 0.0]]],
            vec![line([0.0, 0.0], [2.0, 0.0])],
            0.1,
            limits()
        ));
    }

    #[test]
    fn tolerance_is_inclusive_at_binary_float_boundary() {
        let paths = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        let spans = vec![line([0.0, 0.125], [1.0, 0.125])];
        assert!(certify(paths.clone(), spans.clone(), 0.125, limits()));
        assert!(!certify(paths, spans, 0.125_f64.next_down(), limits()));
    }

    #[test]
    fn one_global_node_budget_covers_forward_and_reverse() {
        let paths = vec![vec![[0.0, 0.0], [1.0, 0.0]]];
        let spans = vec![line([0.0, 0.0], [1.0, 0.0])];
        assert!(!certify(
            paths.clone(),
            spans.clone(),
            0.0,
            Limits {
                max_nodes: 1,
                ..limits()
            }
        ));
        assert!(certify(
            paths,
            spans,
            0.0,
            Limits {
                max_nodes: 2,
                ..limits()
            }
        ));
    }

    #[test]
    fn depth_zero_allows_direct_proof_but_no_subdivision() {
        let shape = ([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], false);
        let source = vec![vec![[0.0, 0.0], [0.5, 0.75], [1.0, 0.0]]];
        assert!(!certify(
            source.clone(),
            vec![shape],
            0.2,
            Limits {
                max_depth: 0,
                ..limits()
            }
        ));
        assert!(certify(source, vec![shape], 0.2, limits()));
    }

    #[test]
    fn closed_paths_preserve_the_explicit_closing_segment() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]];
        let spans = points.windows(2).map(|p| line(p[0], p[1])).collect();
        assert!(certify(vec![points], spans, 0.0, limits()));
    }

    #[test]
    fn indexed_and_linear_witness_order_agree() {
        let source: Vec<_> = (0..=64).map(|i| [f64::from(i) / 8.0, 0.0]).collect();
        let spans = vec![line([0.0, 0.0], [8.0, 0.0])];
        let indexed = certify(
            vec![source.clone()],
            spans.clone(),
            0.0,
            Limits {
                index_min_segments: 1,
                ..limits()
            },
        );
        let linear = certify(
            vec![source],
            spans,
            0.0,
            Limits {
                index_min_segments: usize::MAX,
                ..limits()
            },
        );
        assert!(indexed && linear);
    }

    #[test]
    fn overflowing_float_envelopes_fall_back_to_exact_predicates() {
        let huge = f64::MAX;
        assert!(certify(
            vec![vec![[-huge, 0.0], [huge, 0.0]]],
            vec![line([-huge, 0.0], [huge, 0.0])],
            0.0,
            Limits {
                index_min_segments: 1,
                ..limits()
            }
        ));
    }
}

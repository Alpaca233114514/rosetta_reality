import numpy as np

from scripts.diagnose_hestia_geometric_support import convex_hull, inside_hull, nearest_geometry


def test_axis_range_does_not_imply_convex_support_and_boundary_is_included():
    train = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.2]])
    hull = convex_hull(train)
    assert inside_hull(np.array([[0.8, 0.8], [0.5, 0.5], [0.0, 0.0]]), hull).tolist() == [
        False,
        True,
        True,
    ]


def test_development_extremes_cannot_change_train_scaling_or_loo_neighbors():
    features = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [0.2, 0.3], [5.0, 6.0]])
    donors, distances, scaling = nearest_geometry(features, 3)
    features[-1] = 1e9
    other, other_distances, other_scaling = nearest_geometry(features, 3)
    assert scaling == other_scaling
    np.testing.assert_array_equal(donors[:-1], other[:-1])
    np.testing.assert_array_equal(distances[:-1], other_distances[:-1])


def test_leave_one_out_never_selects_self_and_ties_follow_fixed_order():
    features = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    donors, _, _ = nearest_geometry(features, 3)
    assert donors.tolist() == [1, 0, 0, 0]

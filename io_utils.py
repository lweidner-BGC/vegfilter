import numpy as np
import laspy


def load_las(path) -> tuple:
    """Load a .las/.laz file. Returns (xyz float32 (N,3), LasData)."""
    las = laspy.read(path)
    xyz = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    return xyz, las


def rgb_to_lab_ab(las) -> tuple:
    """Compute CIE LAB a* and b* channels from LAS uint16 RGB.

    Returns (a, b) as float32 arrays of shape (N,).
    a* = green↔red axis, b* = blue↔yellow axis.
    Neither encodes brightness, so the result is illumination-independent.
    """
    r = np.asarray(las.red,   dtype=np.float32) / 65535.0
    g = np.asarray(las.green, dtype=np.float32) / 65535.0
    b = np.asarray(las.blue,  dtype=np.float32) / 65535.0

    # sRGB → linear
    def _lin(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    rl, gl, bl = _lin(r), _lin(g), _lin(b)

    # Linear RGB → XYZ (D65)
    X = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl
    Y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    Z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl

    # Normalise by D65 white point
    X /= 0.95047
    Z /= 1.08883  # Y already normalised by 1.0

    def _f(t):
        return np.where(t > 0.008856, t ** (1.0 / 3.0), 7.787 * t + 16.0 / 116.0)

    fx, fy, fz = _f(X), _f(Y), _f(Z)
    a = (500.0 * (fx - fy)).astype(np.float32)
    b_ch = (200.0 * (fy - fz)).astype(np.float32)
    return a, b_ch


_LAB_FIELDS = {"lab_a", "lab_b"}


def get_scalar_field(las, field_name) -> np.ndarray:
    """Extract a named scalar field from LasData as float32.

    Checks standard dimensions first, then extra dims.
    Raises KeyError with available field names if not found.
    """
    # Standard dimensions
    if hasattr(las, field_name):
        try:
            return np.asarray(getattr(las, field_name), dtype=np.float32)
        except Exception:
            pass

    # Extra dimensions
    extra_names = list(las.point_format.extra_dimension_names)
    if field_name in extra_names:
        return np.asarray(las[field_name], dtype=np.float32)

    available = list(las.point_format.dimension_names) + extra_names
    raise KeyError(
        f"Field '{field_name}' not found. Available fields: {available}"
    )


def get_labels(las, label_field="Classification") -> np.ndarray:
    """Return integer labels from named scalar field.

    Float NaN values (unlabeled points stored as NaN instead of -1)
    are mapped to -1 before casting to int32.
    """
    raw = np.asarray(getattr(las, label_field))
    if np.issubdtype(raw.dtype, np.floating):
        raw = np.where(np.isnan(raw), -1.0, raw)
    return raw.astype(np.int32)


def write_las_with_prediction(
    source_las,
    predictions,
    output_path,
    field_name="PredictedClass",
    probabilities=None,
    prob_field_name="VegProbability",
):
    """Write a LasData copy with prediction (and optionally probability) extra dims."""
    extra_dims = [laspy.ExtraBytesParams(name=field_name, type=np.int32)]
    if probabilities is not None:
        extra_dims.append(laspy.ExtraBytesParams(name=prob_field_name, type=np.float32))

    header = laspy.LasHeader(
        version=source_las.header.version,
        point_format=source_las.point_format.id,
    )
    header.add_extra_dims(extra_dims)
    header.offsets = source_las.header.offsets
    header.scales = source_las.header.scales
    out = laspy.LasData(header=header)

    # Copy all standard + original extra dimensions
    for dim in source_las.point_format.dimension_names:
        try:
            out[dim] = source_las[dim]
        except Exception:
            pass

    out[field_name] = predictions.astype(np.int32)
    if probabilities is not None:
        out[prob_field_name] = probabilities.astype(np.float32)

    out.write(output_path)

"""Compatibility helpers for torchvision datapoints/tv_tensors."""

try:
    from torchvision import datapoints as datapoints
except ImportError:
    from torchvision import tv_tensors

    if not hasattr(tv_tensors.BoundingBoxes, "spatial_size"):
        tv_tensors.BoundingBoxes.spatial_size = property(lambda self: self.canvas_size)

    class _BoundingBox(tv_tensors.BoundingBoxes):
        def __new__(cls, data, *, format, spatial_size, **kwargs):
            return super().__new__(cls, data, format=format, canvas_size=spatial_size, **kwargs)

    class _DataPointsCompat:
        Image = tv_tensors.Image
        Video = tv_tensors.Video
        Mask = tv_tensors.Mask
        BoundingBox = _BoundingBox
        BoundingBoxes = tv_tensors.BoundingBoxes
        BoundingBoxFormat = tv_tensors.BoundingBoxFormat

    datapoints = _DataPointsCompat()

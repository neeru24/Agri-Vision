import numpy as np
from hypothesis import given, settings, strategies as st
from app import preprocess_image_for_resnet

# Generate constant-color RGB image arrays
# Heights and widths from 1 to 500 pixels
@given(
    st.lists(
        st.integers(min_value=0, max_value=255),
        min_size=3, max_size=3
    ).flatmap(lambda rgb: st.builds(
        lambda h, w: np.full((h, w, 3), rgb, dtype=np.uint8),
        st.integers(min_value=1, max_value=500),
        st.integers(min_value=1, max_value=500)
    ))
)
@settings(max_examples=50, deadline=1000)
def test_preprocess_image_for_resnet_property(image_array):
    """
    Test that preprocess_image_for_resnet always returns a tensor
    of shape (1, 3, 224, 224) regardless of the input image dimensions.
    """
    tensor = preprocess_image_for_resnet(image_array)
    assert tensor.shape == (1, 3, 224, 224)

# Generate constant-color grayscale image arrays
# Heights and widths from 1 to 500 pixels
@given(
    st.lists(
        st.integers(min_value=0, max_value=255),
        min_size=1, max_size=1
    ).flatmap(lambda gray: st.builds(
        lambda h, w: np.full((h, w), gray[0], dtype=np.uint8),
        st.integers(min_value=1, max_value=500),
        st.integers(min_value=1, max_value=500)
    ))
)
@settings(max_examples=20, deadline=1000)
def test_preprocess_image_for_resnet_grayscale(image_array):
    """
    Test how the preprocessing handles grayscale images (2 dimensions).
    It should convert grayscale to 3 channels and return (1, 3, 224, 224).
    """
    tensor = preprocess_image_for_resnet(image_array)
    assert tensor.shape == (1, 3, 224, 224)

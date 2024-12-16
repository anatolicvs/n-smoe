import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import chex
import flax
import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.nn.initializers import _compute_fans

class SpectralNormalizedParameter(flax.struct.PyTreeNode, nn.meta.AxisMetadata):
    value: chex.Array
    init_scale: float = flax.struct.field(pytree_node=False) # Only used for debugging.
    init_ratio: float = flax.struct.field(pytree_node=False) # Only used for debugging.
    lr_scale: float = flax.struct.field(pytree_node=False)
    def unbox(self):
        return self.value
    def replace_boxed(self, value):
        return self.replace(value=value)
    def add_axis(self, index, params):
        return self
    def remove_axis(self, index, params):
        return self

def spectral_init(init_scale=1, lr_scale=1, in_axis=-2, out_axis=-1, batch_axis=(), dtype=jnp.float_):
    def init(key, shape: core.Shape, dtype):
        dtype = dtypes.canonicalize_dtype(dtype)
        named_shape = core.as_named_shape(shape)
        if len(shape) == 1: # Bias
            fan_out = shape[0]
            fan_in = 1
        elif len(shape) == 4: # Conv
            fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)
            fan_out = shape[-1]
        else:
            fan_in, fan_out = _compute_fans(named_shape, in_axis, out_axis, batch_axis)

        # Spectral Scaling
        scale = np.sqrt(fan_out / fan_in) * (1 / (np.sqrt(fan_in) + np.sqrt(fan_out)))
        lr = 1/fan_in

        # DEBUG
        # scale = 1/np.sqrt(fan_in)
        # lr = 1
        # END DEBUG

        ratio = scale / (1/np.sqrt(fan_in)) # Ratio between computed_scale and LeCun scale.
        lr = lr * lr_scale
        scale = scale * init_scale
        scale = scale / np.array(.87962566103423978) # Scale by truncated normal constant.
        param = jax.random.truncated_normal(key, -2, 2, shape, dtype) * scale


        param = SpectralNormalizedParameter(value=param, init_scale=scale, init_ratio=ratio, lr_scale=lr)
        return param
    return init

def scale_spectral_norm():
    def init_fn(params):
        return optax.EmptyState()
    def update_fn(updates, state, params=None):
        del params
        def scale_updates(update): # update is either a jax array or a SpectralNormalizedParameter.
            if isinstance(update, SpectralNormalizedParameter):
                return nn.meta.replace_boxed(update, nn.meta.unbox(update) * update.lr_scale)
            return update
        updates = jax.tree_util.tree_map(scale_updates, updates, is_leaf=lambda leaf: isinstance(leaf, SpectralNormalizedParameter))
        return updates, state
    return optax.GradientTransformation(init_fn, update_fn)


def conv_macs(h, w, c_in, c_out, k=3):
    """Multiply-accumulates of one k x k convolution on an h x w feature map."""
    return h * w * c_in * c_out * k * k

s, H, W, C = 4, 128, 128, 64                    # x4 SR of a 128 x 128 input, 64 feature channels
body_pre = conv_macs(s * H, s * W, C, C)        # one body layer at HR size (pre-upsampling)
body_post = conv_macs(H, W, C, C)               # the same layer at LR size (post-upsampling)
tail = conv_macs(H, W, C, 3 * s * s)            # sub-pixel tail: 64 -> 3*s*s channels, then pixel shuffle
print(f"one body layer, pre-upsampling : {body_pre / 1e9:.2f} GMAC")
print(f"one body layer, post-upsampling: {body_post / 1e9:.2f} GMAC (ratio {body_pre / body_post:.0f})")
print(f"sub-pixel tail, 64 -> 48 channels at LR size: {tail / 1e9:.2f} GMAC")

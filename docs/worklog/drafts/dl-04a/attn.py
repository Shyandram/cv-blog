C, M = 180, 8
for side in (64, 128):
    hw = side * side
    g = 4 * hw * C * C + 2 * hw * hw * C
    w = 4 * hw * C * C + 2 * M * M * hw * C
    print(side, f"global {g / 1e9:.1f} GMAC  window {w / 1e9:.2f} GMAC  ratio {g / w:.1f}")

"""Prompt pairs for CLAP-guided string shaping; each is run in both directions."""
PAIRS = {
    'metallic_dull': ('a metallic plucked steel string', 'a dull, muted nylon string'),
    'bright_warm': ('a bright, shimmering plucked string', 'a warm, mellow plucked string'),
    'sustain_muted': ('a plucked string with a long, ringing sustain', 'a short, muted plucked string that stops quickly'),
    'thin_thick': ('a thin, tinny plucked string', 'a thick, deep, heavy plucked bass string'),
}


def directions(pair):
    """Two (target, contrast) tuples for a pair key."""
    a, b = PAIRS[pair]
    return [(a, b), (b, a)]

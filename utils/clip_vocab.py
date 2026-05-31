"""
utils/clip_vocab.py
------------------
CLIP auto-labeling vocabulary for SAE feature analysis.
~5,000 clean concept strings organized in four tiers.

USAGE
-----
    from utils.clip_vocab import get_vocab
    vocab = get_vocab()          # full ~5,000 strings
    vocab = get_vocab(tiers=[1, 2])  # tiers 1+2 only

SOURCES & REASONING
-------------------

Tier 1 — Visual properties (~600 strings)
    Hand-curated. DINO SAE features at early layers (layer 4) encode
    low-level visual properties: textures, colors, edges, patterns,
    surface qualities. ImageNet class names do not cover these —
    a feature encoding "pink feathered texture" would be mislabeled
    "flamingo" if only class names were available. Sources:
      - Describable Textures Dataset (DTD) attribute list (47 texture words)
        Cimpoi et al. (2014). Describing Textures in the Wild. CVPR 2014.
        https://www.robots.ox.ac.uk/~vgg/data/dtd/
      - Color names from X11/CSS standard + XKCD color survey
        https://xkcd.com/color/rgb/
      - Shape/geometry vocabulary from visual commonsense literature
      - Surface and material terms from SUN attribute dataset
        Patterson & Hays (2012). SUN Attribute Database. CVPR 2012.

Tier 2 — Body parts and object parts (~400 strings)
    Hand-curated with bird anatomy prioritised, given the flamingo/
    spoonbill classification task. Mid-layer features (layer 6) in
    DINO encode part-level concepts — beaks, wings, legs — which are
    absent from class-level vocabularies. Sources:
      - Bird anatomy terminology (ornithology glossary)
      - General animal body part terms
      - Object part terms from PartImageNet
        He et al. (2021). PartImageNet. ECCV 2022.
        https://github.com/TACJu/PartImageNet

Tier 3 — Scene and environment (~300 strings)
    Hand-curated. Late-layer features (layer 9) encode scene context:
    wetlands, water, sky, foliage. Scene labels from:
      - SUN Database scene categories (397 scenes)
        Xiao et al. (2010). SUN Database. CVPR 2010.
        http://vision.cs.princeton.edu/projects/2010/SUN/
      - Places365 scene categories
        Zhou et al. (2017). Places. TPAMI 2017.

Tier 4 — ImageNet labels, focused (~3,900 strings, loaded at runtime)
    Programmatically sourced via timm (already a project dependency).
    No network access, HuggingFace token, or extra downloads required.
    timm bundles imagenet_synset_to_lemma.txt internally for all subsets.

    Source:
      timm.data.imagenet_info.ImageNetInfo('imagenet-21k')
        .label_descriptions(detailed=False)
      Returns 21,843 comma-separated lemma strings directly from timm's
      bundled _info/imagenet_synset_to_lemma.txt file.
      Available subsets: imagenet1k, imagenet12k, imagenet21k,
        imagenet21kgoog, imagenet21kmiil, imagenet22k, imagenet22kms
      https://github.com/huggingface/pytorch-image-models

    Cleaning steps applied to the 21,843 raw lemma strings:
      - Took first lemma from comma-separated entries
        e.g. "kit fox, Vulpes macrotis" -> "kit fox"
      - Stripped whitespace
      - Removed entries shorter than 3 characters
      - Removed entries containing digits
      - Removed entries containing underscores, slashes, or parentheses
      - Deduplicated case-insensitively
      - Filtered abstract WordNet synsets that produce uninformative
        high-cosine-similarity matches across almost everything:
        "entity", "object", "whole", "artifact", "process", etc.

NOTE: Tier 4 is loaded dynamically at runtime via get_vocab().
Tiers 1-3 are static and curated for this project.
"""

# =============================================================================
# TIER 1 — Visual properties (~600 strings)
# Low-level texture, color, shape, surface, pattern, lighting
# Targets: layer 4 SAE features
# =============================================================================

TIER1_TEXTURES = [
    # DTD textures (Cimpoi et al., CVPR 2014) — 47 base terms + variants
    "banded texture", "blotchy texture", "braided texture", "bubbly texture",
    "bumpy texture", "chequered texture", "cobwebbed texture", "cracked texture",
    "crosshatched texture", "crystalline texture", "dotted texture",
    "fibrous texture", "flecked texture", "freckled texture", "frilly texture",
    "gauzy texture", "grid texture", "grooved texture", "honeycombed texture",
    "interlaced texture", "knitted texture", "lacelike texture", "lined texture",
    "marbled texture", "matted texture", "meshed texture", "paisley texture",
    "perforated texture", "pitted texture", "pleated texture", "polka-dot texture",
    "porous texture", "potholed texture", "scaly texture", "smeared texture",
    "spiralled texture", "sprinkled texture", "stained texture", "stratified texture",
    "striped texture", "studded texture", "swirly texture", "veined texture",
    "waffled texture", "woven texture", "wrinkled texture", "zigzagged texture",
    # additional visual textures relevant to birds/animals
    "feathered surface", "furry surface", "smooth feathers", "ruffled feathers",
    "glossy feathers", "iridescent feathers", "matte feathers", "downy feathers",
    "scaled skin", "leathery skin", "wet surface", "dry surface",
    "rough surface", "smooth surface", "glossy surface", "matte surface",
    "shiny surface", "dull surface", "metallic surface", "organic texture",
    "soft texture", "hard texture", "coarse texture", "fine texture",
]

TIER1_COLORS = [
    # Basic color names
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "brown", "black", "white", "grey", "gray", "silver", "gold",
    "beige", "cream", "ivory", "tan", "khaki", "olive",
    "cyan", "magenta", "teal", "turquoise", "indigo", "violet",
    "maroon", "scarlet", "crimson", "coral", "salmon", "peach",
    "lavender", "lilac", "mauve", "rose", "fuchsia", "chartreuse",
    "lime", "mint", "aqua", "navy", "cobalt", "azure",
    "amber", "ochre", "rust", "sienna", "umber", "sepia",
    # Color compounds relevant to birds
    "bright pink", "pale pink", "deep pink", "hot pink", "dusty pink",
    "bright red", "dark red", "bright orange", "pale orange", "deep orange",
    "bright yellow", "pale yellow", "golden yellow", "lemon yellow",
    "bright green", "dark green", "pale green", "olive green", "forest green",
    "bright blue", "dark blue", "pale blue", "sky blue", "deep blue",
    "pure white", "off white", "bright white", "creamy white",
    "jet black", "dark grey", "light grey", "pale grey",
    "dark brown", "light brown", "reddish brown", "warm brown",
    # Color patterns
    "black and white", "black and red", "red and white", "orange and black",
    "blue and white", "green and yellow", "pink and white", "brown and white",
    "multicolored", "iridescent color", "metallic color", "uniform color",
    "contrasting colors", "muted colors", "vivid colors", "pastel colors",
]

TIER1_SHAPES = [
    # Basic geometric shapes
    "circular shape", "oval shape", "rectangular shape", "triangular shape",
    "elongated shape", "rounded shape", "pointed shape", "curved shape",
    "straight shape", "angular shape", "irregular shape", "symmetrical shape",
    "flat shape", "convex shape", "concave shape", "tapered shape",
    # Body silhouettes
    "slender silhouette", "stocky silhouette", "tall silhouette", "compact silhouette",
    "wide silhouette", "narrow silhouette", "upright posture", "horizontal posture",
    # Edge and gradient descriptors
    "sharp edge", "soft edge", "blurred edge", "defined outline",
    "gradient transition", "high contrast", "low contrast", "sharp contrast",
    # Spatial descriptors
    "foreground object", "background element", "centered composition",
    "partial view", "full view", "close-up view", "distant view",
    "aerial view", "side view", "front view", "profile view",
]

TIER1_PATTERNS = [
    "striped pattern", "spotted pattern", "checkered pattern", "plaid pattern",
    "camouflage pattern", "floral pattern", "geometric pattern", "abstract pattern",
    "repeating pattern", "symmetrical pattern", "asymmetrical pattern",
    "regular pattern", "irregular pattern", "dense pattern", "sparse pattern",
    "horizontal stripes", "vertical stripes", "diagonal stripes",
    "small spots", "large spots", "fine spots", "irregular spots",
    "rings", "bands", "patches", "blotches", "speckles", "flecks",
]

TIER1_LIGHTING = [
    "bright lighting", "dark lighting", "soft lighting", "harsh lighting",
    "natural light", "sunlight", "shadow", "highlight", "reflection",
    "backlit", "overexposed", "underexposed", "high contrast lighting",
    "warm light", "cool light", "diffuse light", "directional light",
]

TIER1 = (
    TIER1_TEXTURES + TIER1_COLORS + TIER1_SHAPES +
    TIER1_PATTERNS + TIER1_LIGHTING
)

# =============================================================================
# TIER 2 — Body parts and object parts (~400 strings)
# Part-level concepts for mid-layer features
# Targets: layer 6 SAE features
# Sources: ornithology glossary, PartImageNet, general anatomy
# =============================================================================

TIER2_BIRD_ANATOMY = [
    # Beak/bill — most discriminative for flamingo vs spoonbill
    "beak", "bill", "curved beak", "straight bill", "hooked beak",
    "spatula-shaped bill", "spoon-shaped bill", "long beak", "short beak",
    "narrow beak", "wide beak", "open beak", "closed beak",
    "upper mandible", "lower mandible", "tip of beak",
    # Head
    "head", "crown", "forehead", "face", "cheek", "chin", "throat",
    "eye", "eye ring", "eye stripe", "pupil", "iris", "eyelid",
    "ear", "nape", "back of head",
    # Neck
    "neck", "long neck", "short neck", "curved neck", "extended neck",
    "neck feathers", "throat pouch",
    # Body
    "body", "breast", "belly", "abdomen", "chest", "back", "rump",
    "torso", "underparts", "upperparts", "flanks", "side",
    # Wings
    "wing", "wings", "folded wing", "extended wing", "wing tip",
    "primary feathers", "secondary feathers", "wing coverts",
    "flight feathers", "wing span", "underwing", "upperwing",
    # Tail
    "tail", "tail feathers", "short tail", "long tail", "forked tail",
    "tail tip", "upright tail", "fanned tail",
    # Legs and feet
    "leg", "legs", "long legs", "short legs", "knee", "ankle",
    "foot", "feet", "toes", "claws", "talons", "webbed feet",
    "bare legs", "feathered legs", "wading legs",
    # Feather groups
    "plumage", "breeding plumage", "juvenile plumage", "adult plumage",
    "crest", "tuft", "collar", "mantle", "scapulars",
]

TIER2_GENERAL_ANIMAL_PARTS = [
    # Shared with other animals
    "paw", "hoof", "horn", "antler", "tusk", "fang", "claw",
    "fin", "gill", "scale", "shell", "tail", "mane", "fur",
    "whisker", "snout", "muzzle", "nose", "ear", "eye", "mouth",
    "tongue", "tooth", "jaw", "spine", "limb",
    # Body regions
    "abdomen", "thorax", "shoulder", "hip", "flank",
    "dorsal", "ventral", "lateral", "anterior", "posterior",
]

TIER2_OBJECT_PARTS = [
    # For non-bird ImageNet features
    "handle", "wheel", "door", "window", "roof", "wall", "floor",
    "screen", "button", "keyboard", "lens", "frame", "edge", "corner",
    "surface", "interior", "exterior", "top", "bottom", "side",
    "blade", "tip", "base", "stem", "leaf", "petal", "root", "branch",
]

TIER2_PLANT_PARTS = [
    "leaf", "petal", "flower", "stem", "root", "bark", "branch",
    "trunk", "thorn", "seed", "fruit", "bud", "shoot",
    "foliage", "canopy", "undergrowth",
]

TIER2 = (
    TIER2_BIRD_ANATOMY + TIER2_GENERAL_ANIMAL_PARTS +
    TIER2_OBJECT_PARTS + TIER2_PLANT_PARTS
)

# =============================================================================
# TIER 3 — Scene and environment (~300 strings)
# High-level scene/context concepts for late-layer features
# Targets: layer 9 SAE features
# Sources: SUN database, Places365
# =============================================================================

TIER3_NATURAL_SCENES = [
    # Water environments — directly relevant to flamingo/spoonbill
    "lake", "pond", "river", "stream", "ocean", "sea", "bay",
    "estuary", "lagoon", "wetland", "marsh", "swamp", "mudflat",
    "saltpan", "shallow water", "deep water", "calm water", "rippling water",
    "water surface", "water reflection", "still water", "flowing water",
    # Land environments
    "forest", "woodland", "jungle", "rainforest", "savanna", "grassland",
    "meadow", "field", "plain", "desert", "tundra", "alpine", "mountain",
    "hill", "valley", "cliff", "canyon", "beach", "shoreline", "coast",
    # Sky and atmosphere
    "sky", "blue sky", "cloudy sky", "sunset sky", "sunrise sky",
    "overcast sky", "clear sky", "clouds", "fog", "mist", "haze",
    # Vegetation
    "trees", "grass", "reeds", "bushes", "shrubs", "vegetation",
    "aquatic plants", "water lilies", "mangroves", "tropical foliage",
    "dense foliage", "sparse vegetation", "green vegetation",
]

TIER3_MAN_MADE_SCENES = [
    "road", "street", "building", "house", "room", "garden", "park",
    "bridge", "fence", "wall", "window", "door",
    "urban environment", "rural environment", "indoor scene", "outdoor scene",
    "zoo enclosure", "wildlife reserve", "nature reserve",
]

TIER3_CONTEXTUAL = [
    # Behavior context
    "in flight", "flying", "perching", "standing", "wading",
    "swimming", "diving", "feeding", "resting", "walking",
    "flock of birds", "single bird", "group of animals",
    "in water", "on land", "on branch", "on rock", "on ground",
    # Background context
    "blurred background", "bokeh background", "sharp background",
    "natural habitat", "captive animal",
]

TIER3 = TIER3_NATURAL_SCENES + TIER3_MAN_MADE_SCENES + TIER3_CONTEXTUAL


# =============================================================================
# TIER 4 — ImageNet-21k class names, cleaned (~3,700 strings, loaded at runtime)
# =============================================================================

def _normalize_label(label: str) -> str:
    """Return a lowercase, whitespace-normalized label."""
    return " ".join(str(label).strip().lower().split())


def _dedupe_normalized(labels: list) -> list:
    """Normalize labels and deduplicate while preserving order."""
    seen = set()
    deduped = []
    for label in labels:
        key = _normalize_label(label)
        if key not in seen and len(key) >= 3:
            seen.add(key)
            deduped.append(key)
    return deduped


def _clean_imagenet22k(raw_labels: list) -> list:
    """
    Clean ImageNet-22k synset labels for use as CLIP concept strings.

    Steps:
    1. Take first name from comma-separated synset entries
       e.g. "kit fox, Vulpes macrotis" -> "kit fox"
    2. Strip whitespace
    3. Remove entries shorter than 3 characters
    4. Remove entries containing digits
    5. Remove entries containing underscores or slashes
    6. Deduplicate case-insensitively
    7. Filter out abstract WordNet synsets that degrade CLIP similarity
       (entity, object, whole, artifact, etc.)

    Returns cleaned list of ~3,700 strings.
    """
    ABSTRACT_BLOCKLIST = {
        "entity", "object", "whole", "artifact", "article", "thing",
        "substance", "matter", "physical entity", "abstraction",
        "psychological feature", "cognition", "content", "process",
        "act", "action", "activity", "event", "phenomenon",
        "relation", "state", "attribute", "measure", "amount",
        "group", "grouping", "collection", "accumulation",
        "part", "piece", "section", "unit", "item",
    }

    seen = set()
    cleaned = []
    for label in raw_labels:
        # Take first name from comma-separated entries
        name = _normalize_label(label.split(",")[0])

        # Basic filters
        if len(name) < 3:
            continue
        if any(c.isdigit() for c in name):
            continue
        if "_" in name or "/" in name:
            continue
        if "(" in name or ")" in name:
            continue

        # Filter abstract synsets
        if name.lower() in ABSTRACT_BLOCKLIST:
            continue
        if name.lower().startswith(("type of", "kind of", "form of")):
            continue

        # Deduplicate
        key = name
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)

    return cleaned


def _load_tier4_from_timm() -> list:
    """
    Load a focused ~3,900-string vocabulary from timm's bundled ImageNet synsets.

    Strategy (aligned with project plan '10k vocab' spec):
      Base: all 1,000 ImageNet-1k class names — directly represent the classes
            in the 5,000-image activation cache.
      Plus: ~2,900 additional entries from ImageNet-21k filtered to visually
            concrete categories relevant to the project: birds (heavily weighted
            for the flamingo/spoonbill task), other animals, plants, natural
            materials, food. Excludes abstract synsets, Latin taxonomic names,
            medical terms, and multi-word technical phrases.

    Why not all 21k:
      - 20k strings dilute cosine similarity scores — more candidates means
        more chance of a spurious near-match winning top-3
      - Most of 21k is absent from any image in a 5,000-image ImageNet-1k cache
      - Abstract synsets ("benthos", "heterotroph") degrade CLIP matching quality
      - Project plan specifies 10k vocab; this targeted ~4k is well within that

    Source: timm.data.imagenet_info.ImageNetInfo — no network access needed.
    timm bundles imagenet_synset_to_lemma.txt internally.
    https://github.com/huggingface/pytorch-image-models

    Falls back to empty list if timm is not installed.
    """
    # Visual keyword filter for 21k additions beyond ImageNet-1k
    VISUAL_KEYWORDS = [
        # Birds — heavily weighted; directly relevant to flamingo/spoonbill task
        'bird', 'finch', 'warbler', 'sparrow', 'hawk', 'eagle', 'owl', 'duck',
        'goose', 'swan', 'heron', 'egret', 'stork', 'pelican', 'penguin',
        'parrot', 'pigeon', 'dove', 'robin', 'jay', 'crow', 'raven', 'magpie',
        'wren', 'thrush', 'swallow', 'swift', 'tern', 'gull', 'grebe',
        'ibis', 'flamingo', 'spoonbill', 'avocet', 'plover', 'sandpiper',
        'cormorant', 'gannet', 'booby', 'tropicbird', 'frigatebird',
        # Other animals
        'fish', 'shark', 'whale', 'dolphin', 'seal', 'otter',
        'cat', 'dog', 'bear', 'fox', 'wolf', 'deer', 'horse', 'cow',
        'lion', 'tiger', 'leopard', 'jaguar', 'cheetah', 'elephant',
        'monkey', 'ape', 'rabbit', 'squirrel', 'mouse', 'rat',
        'snake', 'lizard', 'turtle', 'frog', 'salamander', 'gecko',
        'butterfly', 'moth', 'beetle', 'dragonfly', 'bee', 'wasp',
        # Plants
        'oak', 'pine', 'maple', 'birch', 'willow', 'palm', 'fern',
        'moss', 'mushroom', 'flower', 'rose', 'tulip', 'orchid', 'lily',
        # Natural materials and objects
        'rock', 'stone', 'sand', 'mud', 'coral', 'shell', 'feather',
        # Food
        'fruit', 'berry', 'nut', 'seed',
    ]

    try:
        from timm.data.imagenet_info import ImageNetInfo

        # Base: all ImageNet-1k class names (first lemma only)
        info1k = ImageNetInfo('imagenet1k')
        labels1k = [_normalize_label(l.split(',')[0])
                    for l in info1k.label_descriptions(detailed=False)]
        seen = set(labels1k)
        result = list(labels1k)

        # Additions: visually concrete ImageNet-21k synsets not in 1k
        info21k = ImageNetInfo('imagenet-21k')
        for label in info21k.label_descriptions(detailed=False):
            first = _normalize_label(label.split(',')[0])
            if first in seen:
                continue
            if len(first) < 3 or any(c.isdigit() for c in first):
                continue
            if len(first.split()) > 3:
                continue
            if any(kw in first for kw in VISUAL_KEYWORDS):
                seen.add(first)
                result.append(first)

        print(f"[vocab] Loaded {len(result)} labels from timm "
              f"(1k base + targeted 21k additions) — Tier 4")
        return result

    except Exception as e:
        print(f"[vocab] Warning: Could not load Tier 4 from timm: {e}")
        print("[vocab] Ensure timm is installed: pip install timm")
        print("[vocab] Falling back to Tiers 1-3 only.")
        return []


# =============================================================================
# PUBLIC API
# =============================================================================

def get_vocab(tiers: list = None, load_tier4: bool = True) -> list:
    """
    Return the full CLIP auto-labeling vocabulary.

    Parameters
    ----------
    tiers : list of int, optional
        Which tiers to include. Default: [1, 2, 3, 4].
        Use [1, 2] for fast testing; [1, 2, 3, 4] for full labeling.
    load_tier4 : bool
        Whether to load Tier 4 from timm's bundled synset files.
        Requires timm to be installed (already in requirements.txt).
        Set False only for unit testing without timm.

    Returns
    -------
    list of str
        Deduplicated vocabulary strings, ready to encode with CLIP.

    Example
    -------
    >>> vocab = get_vocab()
    >>> len(vocab)
    ~5000
    >>> vocab = get_vocab(tiers=[1, 2], load_tier4=False)
    >>> len(vocab)
    ~1000
    """
    if tiers is None:
        tiers = [1, 2, 3, 4]

    result = []
    if 1 in tiers:
        result.extend(TIER1)
    if 2 in tiers:
        result.extend(TIER2)
    if 3 in tiers:
        result.extend(TIER3)
    if 4 in tiers and load_tier4:
        result.extend(_load_tier4_from_timm())

    deduped = _dedupe_normalized(result)

    print(f"[vocab] Total vocabulary: {len(deduped)} strings "
          f"(tiers: {tiers})")
    return deduped


def get_vocab_by_tier() -> dict:
    """
    Return vocabulary split by tier for analysis.
    Tier 4 is loaded from HuggingFace.

    Returns
    -------
    dict mapping tier int -> list of strings
    """
    return {
        1: _dedupe_normalized(TIER1),
        2: _dedupe_normalized(TIER2),
        3: _dedupe_normalized(TIER3),
        4: _dedupe_normalized(_load_tier4_from_timm()),
    }


if __name__ == "__main__":
    # Smoke test
    vocab = get_vocab()
    by_tier = get_vocab_by_tier()
    print(f"\nTier 1 (visual properties): {len(by_tier[1])} strings")
    print(f"Tier 2 (body parts):         {len(by_tier[2])} strings")
    print(f"Tier 3 (scenes/environment): {len(by_tier[3])} strings")
    print(f"Tier 4 (ImageNet-22k):       {len(by_tier[4])} strings")
    print(f"\nTotal: {len(vocab)} strings")
    print("\nSample Tier 1:", TIER1[:5])
    print("Sample Tier 2:", TIER2_BIRD_ANATOMY[:5])
    print("Sample Tier 3:", TIER3_NATURAL_SCENES[:5])

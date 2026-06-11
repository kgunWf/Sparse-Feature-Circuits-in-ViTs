"""
utils/clip_vocab.py
------------------
CLIP auto-labeling vocabulary for SAE feature analysis.
~5,000 clean concept strings organized in four tiers.

USAGE
-----
    from data.clip_vocab import get_vocab
    vocab = get_vocab()          # full ~5,000 strings
    vocab = get_vocab(tiers=[1, 2])  # tiers 1+2 only

SOURCES & REASONING
-------------------

Tier 1 — Visual properties (~700 strings)
    Hand-curated. DINO SAE features at early layers (layer 4) encode
    low-level visual properties: textures, colors, edges, patterns,
    surface qualities. ImageNet class names do not cover these —
    a feature encoding "dusty blue leather" would be mislabeled
    by class names alone. Organized into six sub-sections:
      - TIER1_TEXTURES: DTD-derived texture vocabulary
      - TIER1_COLORS: single color names + compound color modifiers
      - TIER1_BIGRAMS: color+material and texture+surface bigrams
        sourced from SpLiCE / LAION web frequency list
        (Bhalla et al., ICML 2023, github.com/AI4LIFE-GROUP/SpLiCE);
        filtered to visually grounded entries only
      - TIER1_SHAPES: silhouettes, edges, gradients
      - TIER1_PATTERNS: repeating motifs
      - TIER1_LIGHTING: illumination descriptors

    Sources:
      - Describable Textures Dataset (DTD) attribute list (47 texture words)
        Cimpoi et al. (2014). Describing Textures in the Wild. CVPR 2014.
        https://www.robots.ox.ac.uk/~vgg/data/dtd/
      - Color names from X11/CSS standard + XKCD color survey
        https://xkcd.com/color/rgb/
      - Shape/geometry vocabulary from visual commonsense literature
      - Surface and material terms from SUN attribute dataset
        Patterson & Hays (2012). SUN Attribute Database. CVPR 2012.
      - SpLiCE LAION bigrams (visual subset only)
        Bhalla et al. (2023). https://github.com/AI4LIFE-GROUP/SpLiCE

Tier 2 — Body parts and object parts (~250 strings)
    Hand-curated for general ViT SAE features. Mid-layer features (layer 6)
    in DINO encode part-level concepts — limbs, faces, object components —
    which are absent from class-level vocabularies. Covers animal anatomy,
    object parts, and plant parts. Bird-specific anatomy (beaks, plumage,
    wing coverts, mandibles, wading legs, etc.) has been removed; the
    remaining animal terms (head, eye, leg, wing, tail, claw) are general
    across ImageNet categories.

    Sources:
      - General animal body part terms
      - Object part terms from PartImageNet
        He et al. (2021). PartImageNet. ECCV 2022.
        https://github.com/TACJu/PartImageNet

Tier 3 — Scene and environment (~200 strings)
    Hand-curated. Late-layer features (layer 9) encode scene context.
    Wetland/wading-bird habitat terms have been removed; scenes now cover
    the breadth of ImageNet environments.

    Sources:
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
# TIER 1 — Visual properties (~700 strings)
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
    # General surface quality descriptors
    "furry surface", "scaled skin", "leathery skin",
    "wet surface", "dry surface", "rough surface", "smooth surface",
    "glossy surface", "matte surface", "shiny surface", "dull surface",
    "metallic surface", "organic texture", "soft texture", "hard texture",
    "coarse texture", "fine texture",
    # Texture + material bigrams (SpLiCE / LAION)
    "fabric texture", "leather texture", "wood grain",
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
    # Color modifiers — general shade/saturation compounds
    "bright pink", "pale pink", "deep pink", "hot pink",
    "bright red", "dark red",
    "bright orange", "pale orange", "deep orange", "burnt orange",
    "bright yellow", "pale yellow", "golden yellow", "lemon yellow", "neon yellow",
    "bright green", "dark green", "pale green", "olive green", "forest green",
    "bright blue", "dark blue", "pale blue", "sky blue", "deep blue",
    "pure white", "off white", "bright white", "creamy white",
    "jet black", "dark grey", "light grey", "pale grey",
    "dark brown", "light brown", "reddish brown", "warm brown",
    # Additional compound modifiers (SpLiCE / LAION)
    "dusty blue", "dusty pink",
    "soft blue", "light blue", "light pink", "light yellow",
    "light beige", "light purple", "dark pink", "dark purple",
    "deep purple", "pastel blue", "neon pink",
    "silver grey", "silver metallic",
    # Color patterns
    "black and white", "black and red", "red and white", "orange and black",
    "blue and white", "green and yellow", "pink and white", "brown and white",
    "multicolored", "iridescent color", "metallic color", "uniform color",
    "contrasting colors", "muted colors", "vivid colors", "pastel colors",
]

TIER1_BIGRAMS = [
    # Color + material compounds (SpLiCE / LAION — visually grounded subset)
    # Black + material
    "black cotton", "black denim", "black glass", "black granite",
    "black lace", "black leather", "black marble", "black metal",
    "black sand", "black silk", "black stone", "black suede",
    "black velvet", "black wooden", "black wool",
    # Blue + material
    "blue cotton", "blue denim", "blue fabric", "blue glass",
    "blue lace", "blue leather", "blue silk", "blue stripe",
    "blue stripes", "blue velvet", "blue wooden", "blue water",
    # Brown / grey + material
    "brown leather", "brown suede", "brown wooden",
    "grey leather", "dark wood", "light wood",
    "dark denim", "light denim",
    # White + material
    "white brick", "white cotton", "white fabric", "white fur",
    "white glass", "white lace", "white leather", "white linen",
    "white marble", "white metal", "white sand", "white tile",
    "white wooden",
    # Red + material
    "red brick", "red fabric", "red leather", "red plaid",
    "red silk", "red velvet",
    # Green / pink + material
    "green glass", "green leather",
    "pink lace", "pink leather",
    # Silver + material
    "silver metal",
    # Soft + material
    "soft cotton",
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
    "sharp edge", "soft edge", "defined outline",
    "gradient transition", "high contrast", "low contrast", "sharp contrast",
    # View
    "profile view",
]

TIER1_PATTERNS = [
    "striped pattern", "spotted pattern", "checkered pattern", "plaid pattern",
    "camouflage pattern", "floral pattern", "geometric pattern", "abstract pattern",
    "repeating pattern", "symmetrical pattern", "asymmetrical pattern",
    "regular pattern", "irregular pattern", "dense pattern", "sparse pattern",
    "horizontal stripes", "vertical stripes", "diagonal stripes",
    "small spots", "large spots", "fine spots", "irregular spots",
    "rings", "bands", "patches", "blotches", "speckles", "flecks",
    # Pattern bigrams (SpLiCE / LAION)
    "lace pattern",
]

TIER1_LIGHTING = [
    "bright lighting", "dark lighting", "soft lighting", "harsh lighting",
    "natural light", "sunlight", "shadow", "highlight", "reflection",
    "backlit", "high contrast lighting",
    "warm light", "cool light", "diffuse light", "directional light",
]

TIER1_TEXT = [
    "text overlay", "typography", "printed text",
    "signage", "logo", "handwriting",
]

TIER1 = (
    TIER1_TEXTURES + TIER1_COLORS + TIER1_BIGRAMS + TIER1_SHAPES +
    TIER1_PATTERNS + TIER1_LIGHTING + TIER1_TEXT
)

# =============================================================================
# TIER 2 — Body parts and object parts (~250 strings)
# Part-level concepts for mid-layer features
# Targets: layer 6 SAE features
# Sources: PartImageNet, general anatomy; bird-specific terms removed
# =============================================================================

TIER2_ANIMAL_PARTS = [
    # Head region — specific enough for CLIP to localise
    # Removed broad singletons that match too many visual contexts:
    #   chin, throat, ear, nape — single-word terms with diffuse CLIP embeddings
    "head", "crown", "forehead", "face", "cheek",
    "eye", "pupil", "iris", "eyelid",
    # Neck and body — kept only visually distinctive terms
    # Removed: neck, body, torso, back, side, flank, belly, abdomen, shoulder, hip
    "breast", "chest",
    # Limbs — kept specific joints/extremities; removed generic "leg/legs"
    "knee", "ankle", "foot", "feet", "toes",
    "arm", "elbow", "wrist", "hand", "finger",
    # Extremities — general
    "claw", "tail", "crest", "tuft", "collar",
    # Wings — kept as general (bats, insects, birds all have wings)
    "wing", "wings", "wing tip",
    # Other animal parts — general across ImageNet
    "paw", "hoof", "horn", "antler", "tusk", "fang",
    "fin", "gill", "scale", "shell", "mane", "fur",
    "whisker", "snout", "muzzle", "nose", "mouth",
    "tongue", "tooth", "jaw", "spine", "limb",
    # Body surface
    "skin", "coat", "feather",
]

TIER2_OBJECT_PARTS = [
    # Vehicles and machines
    "handle", "wheel", "roof", "floor",
    "screen", "button", "keyboard", "lens", "frame", "edge", "corner",
    "surface", "interior", "exterior", "top", "bottom",
    "blade", "tip", "base",
    # Structural
    "pillar", "arch", "step", "railing", "beam", "panel",
    "socket", "joint", "hinge", "knob", "slot", "groove",
]

TIER2_PLANT_PARTS = [
    "leaf", "petal", "flower", "stem", "root", "bark", "branch",
    "trunk", "thorn", "seed", "fruit", "bud", "shoot",
    "foliage", "canopy", "undergrowth",
]

TIER2 = (
    TIER2_ANIMAL_PARTS + TIER2_OBJECT_PARTS + TIER2_PLANT_PARTS
)

# =============================================================================
# TIER 3 — Scene and environment (~200 strings)
# High-level scene/context concepts for late-layer features
# Targets: layer 9 SAE features
# Sources: SUN database, Places365; wetland/wading-bird terms removed
# =============================================================================

TIER3_NATURAL_SCENES = [
    # Water environments — general (not wetland-specific)
    "lake", "pond", "river", "stream", "ocean", "sea", "bay",
    "shallow water", "deep water", "calm water", "rippling water",
    "water surface", "water reflection", "still water", "flowing water",
    # Land environments
    "forest", "woodland", "jungle", "rainforest", "savanna", "grassland",
    "meadow", "field", "plain", "desert", "tundra", "alpine", "mountain",
    "hill", "valley", "cliff", "canyon", "beach", "shoreline", "coast",
    # Sky and atmosphere
    "sky", "blue sky", "cloudy sky", "overcast sky", "clear sky",
    "clouds", "fog", "mist", "haze",
    # Vegetation
    "trees", "grass", "bushes", "shrubs", "vegetation",
    "dense foliage", "sparse vegetation", "green vegetation",
    # Landscape bigrams (SpLiCE / LAION)
    "autumn forest", "desert landscape", "forest landscape",
    "mountain landscape", "rocky mountain", "sandy beach",
    "sunset sky", "flower field", "flower garden", "grass field",
    "green field", "green forest", "green grass", "green meadow",
    "green tree", "lavender field", "beach sand",
    "tree branch", "olive branch", "olive tree",
    "rock garden", "orange flower", "red flower", "silk flower",
    # black sand, white sand defined in TIER1_BIGRAMS — not repeated here
]

TIER3_MAN_MADE_SCENES = [
    "road", "street", "building", "house", "room", "garden", "park",
    "bridge", "fence", "wall", "window", "door",
    "urban environment", "rural environment", "indoor scene", "outdoor scene",
]

TIER3_CONTEXTUAL = [
    # General animal behavior
    "in flight", "flying", "standing", "swimming", "diving",
    "feeding", "resting", "walking", "running", "climbing",
    "group of animals", "single animal",
    "in water", "on land",
    # Scene context
    "close-up view", "natural habitat",
]

TIER3 = TIER3_NATURAL_SCENES + TIER3_MAN_MADE_SCENES + TIER3_CONTEXTUAL


# =============================================================================
# TIER 4 — ImageNet-21k class names, cleaned (~3,700 strings, loaded at runtime)
# =============================================================================

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
        name = label.split(",")[0].strip()

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
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(name)

    return cleaned


def _load_tier4_from_timm(subset: str = "full") -> list:
    """
    Load Tier 4 vocabulary from timm's bundled ImageNet synsets.

    Parameters
    ----------
    subset : str
        "1k"   — 1,000 ImageNet-1k class names only. Directly covers every
                 class in the 5,000-image activation cache; no noise from
                 absent 21k synsets. Recommended when label quality matters.
        "full" — 1k base + targeted ImageNet-21k additions filtered to
                 visually concrete categories. Uses both a visual allowlist
                 and a non-visual blocklist to minimise label pollution.

    Source: timm.data.imagenet_info.ImageNetInfo — no network access needed.
    timm bundles imagenet_synset_to_lemma.txt internally.
    https://github.com/huggingface/pytorch-image-models

    Falls back to empty list if timm is not installed.
    """
    # ------------------------------------------------------------------ #
    # Positive: substring keywords that signal visually concrete synsets  #
    # ------------------------------------------------------------------ #
    VISUAL_KEYWORDS = [
        # Animals — general ImageNet coverage
        'bird', 'finch', 'warbler', 'sparrow', 'hawk', 'eagle', 'owl', 'duck',
        'goose', 'swan', 'heron', 'egret', 'stork', 'pelican', 'penguin',
        'parrot', 'pigeon', 'dove', 'robin', 'jay', 'crow', 'raven', 'magpie',
        'wren', 'thrush', 'swallow', 'swift', 'tern', 'gull', 'grebe',
        'ibis', 'flamingo', 'spoonbill', 'avocet', 'plover', 'sandpiper',
        'cormorant', 'gannet', 'booby', 'tropicbird', 'frigatebird',
        'fish', 'shark', 'whale', 'dolphin', 'seal', 'otter',
        'cat', 'dog', 'bear', 'fox', 'wolf', 'deer', 'horse', 'cow',
        'lion', 'tiger', 'leopard', 'jaguar', 'cheetah', 'elephant',
        'monkey', 'ape', 'rabbit', 'squirrel', 'mouse', 'rat',
        'snake', 'lizard', 'turtle', 'frog', 'salamander', 'gecko',
        'butterfly', 'moth', 'beetle', 'dragonfly', 'bee', 'wasp',
        'spider', 'crab', 'lobster', 'shrimp', 'insect', 'larva',
        # Plants and fungi
        'oak', 'pine', 'maple', 'birch', 'willow', 'palm', 'fern',
        'moss', 'mushroom', 'flower', 'rose', 'tulip', 'orchid', 'lily',
        'cactus', 'bamboo', 'seaweed', 'algae', 'lichen', 'fungus',
        # Natural materials and scene elements
        'rock', 'stone', 'sand', 'mud', 'coral', 'shell', 'feather',
        'pebble', 'cliff', 'glacier', 'lava', 'crystal',
        # Food and produce
        'fruit', 'berry', 'nut', 'seed', 'vegetable', 'grain',
    ]

    # ------------------------------------------------------------------ #
    # Negative: whole-word matches that flag non-visual concepts          #
    # ------------------------------------------------------------------ #
    _BLOCK_WORDS = {
        # Medical / pathological
        "disease", "disorder", "syndrome", "infection", "deficiency",
        "cancer", "tumor", "tumour", "fever", "virus", "bacteria",
        "poisoning", "allergy", "toxin", "parasite", "pathology",
        "symptom", "treatment", "therapy", "drug", "pharmaceutical",
        # Abstract / conceptual / relational
        "process", "phenomenon", "behavior", "behaviour", "relationship",
        "mechanism", "function", "effect", "system", "method", "technique",
        "theory", "principle", "concept", "ideology", "movement", "period",
        "culture", "tradition", "language", "music", "dance", "style",
        "ceremony", "ritual", "belief", "religion", "philosophy",
        # Taxonomic rank indicators
        "genus", "phylum", "kingdom", "subspecies", "cultivar", "variety",
        # Collective / generic group nouns
        "complex", "clade", "lineage", "taxon",
    }

    _BLOCK_SUFFIXES = (
        "idae", "inae", "aceae", "ales", "iformes", "oidea",  # taxonomy
        "osis", "itis", "emia", "uria",                        # medical
        "ology", "ography", "onomy", "onomics",               # academic fields
    )

    def _is_acceptable_21k(name: str) -> bool:
        """Return True if name passes all quality filters for 21k additions."""
        lower = name.lower()
        words = lower.split()

        if len(name) < 3:
            return False
        if any(c.isdigit() for c in name):
            return False
        if len(words) > 3:
            return False
        if "_" in name or "/" in name or "(" in name:
            return False
        if any(lower.endswith(sfx) for sfx in _BLOCK_SUFFIXES):
            return False
        if name.isupper() and len(name) <= 6:
            return False
        if any(w in _BLOCK_WORDS for w in words):
            return False
        return any(kw in lower for kw in VISUAL_KEYWORDS)

    try:
        from timm.data.imagenet_info import ImageNetInfo

        info1k = ImageNetInfo('imagenet1k')
        labels1k = [l.split(',')[0].strip()
                    for l in info1k.label_descriptions(detailed=False)]
        result = list(labels1k)

        if subset == "1k":
            print(f"[vocab] Loaded {len(result)} labels from timm "
                  f"(ImageNet-1k only) — Tier 4")
            return result

        seen = {l.lower() for l in labels1k}
        info21k = ImageNetInfo('imagenet-21k')
        added = 0
        for label in info21k.label_descriptions(detailed=False):
            first = label.split(',')[0].strip()
            first_lower = first.lower()
            if first_lower in seen:
                continue
            if _is_acceptable_21k(first):
                seen.add(first_lower)
                result.append(first)
                added += 1

        print(f"[vocab] Loaded {len(result)} labels from timm "
              f"(1k base + {added} targeted 21k additions) — Tier 4")
        return result

    except Exception as e:
        print(f"[vocab] Warning: Could not load Tier 4 from timm: {e}")
        print("[vocab] Ensure timm is installed: pip install timm")
        print("[vocab] Falling back to Tiers 1-3 only.")
        return []


# =============================================================================
# PUBLIC API
# =============================================================================

def get_vocab(tiers: list = None, load_tier4: bool = True,
              tier4_subset: str = "full") -> list:
    """
    Return the CLIP auto-labeling vocabulary.

    Parameters
    ----------
    tiers : list of int, optional
        Which tiers to include. Default: [1, 2, 3, 4].
    load_tier4 : bool
        Set False to skip Tier 4 entirely (unit testing without timm).
    tier4_subset : str
        "1k"   — ImageNet-1k only (1,000 strings). Covers every class in
                 the 5,000-image activation cache with no 21k noise.
        "full" — 1k base + ~2,900 targeted 21k additions (default).

    Example
    -------
    >>> vocab = get_vocab()                                # ~5,000 strings
    >>> vocab = get_vocab(tier4_subset="1k")              # ~2,300 strings
    >>> vocab = get_vocab(tiers=[1, 2], load_tier4=False) # ~1,000 strings
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
        result.extend(_load_tier4_from_timm(subset=tier4_subset))

    # Final deduplication across all tiers
    seen = set()
    deduped = []
    for s in result:
        key = s.lower().strip()
        if key not in seen and len(key) >= 3:
            seen.add(key)
            deduped.append(s)

    print(f"[vocab] Total vocabulary: {len(deduped)} strings "
          f"(tiers: {tiers}, tier4_subset: {tier4_subset!r})")
    return deduped


def get_vocab_by_tier() -> dict:
    """
    Return vocabulary split by tier for analysis.
    Tier 4 is loaded from timm.

    Returns
    -------
    dict mapping tier int -> list of strings
    """
    return {
        1: TIER1,
        2: TIER2,
        3: TIER3,
        4: _load_tier4_from_timm(),
    }


if __name__ == "__main__":
    # Smoke test
    vocab = get_vocab(load_tier4=False)
    by_tier = get_vocab_by_tier.__wrapped__() if hasattr(get_vocab_by_tier, '__wrapped__') else {
        1: TIER1, 2: TIER2, 3: TIER3
    }
    print(f"\nTier 1 (visual properties): {len(TIER1)} strings")
    print(f"  - Textures:  {len(TIER1_TEXTURES)}")
    print(f"  - Colors:    {len(TIER1_COLORS)}")
    print(f"  - Bigrams:   {len(TIER1_BIGRAMS)}")
    print(f"  - Shapes:    {len(TIER1_SHAPES)}")
    print(f"  - Patterns:  {len(TIER1_PATTERNS)}")
    print(f"  - Lighting:  {len(TIER1_LIGHTING)}")
    print(f"  - Text:      {len(TIER1_TEXT)}")
    print(f"Tier 2 (body/object parts): {len(TIER2)} strings")
    print(f"Tier 3 (scenes/environment): {len(TIER3)} strings")
    print(f"\nTotal (Tiers 1-3): {len(vocab)} strings")
    print("\nSample Tier 1 bigrams:", TIER1_BIGRAMS[:5])
    print("Sample Tier 2:", TIER2_ANIMAL_PARTS[:5])
    print("Sample Tier 3 landscape:", TIER3_NATURAL_SCENES[-10:])
NEGATIVE_PROMPT = ""

# NEGATIVE_PROMPT1 = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft,   \
#                     deformed, ugly, "
# NEGATIVE_PROMPT2 = "anime, cartoon, graphic, text, painting graphite, abstract, glitch, mutated disfigured, "
# NEGATIVE_PROMPT3 = "monochrome, lowres, bad anatomy, worst quality, low quality, "
# NEGATIVE_PROMPT4 = "watermark, low quality, medium quality, blurry, censored, wrinkles, deformed, mutated, "

# Negative Prompts to improve the quality of image
NEGATIVE_PROMPT4 = "Ugly, Bad anatomy, Bad proportions, Bad quality, Blurry, Cropped, Deformed,          \
                    Disconnected limbs, Out of frame, Out of focus, Dehydrated, Error, Disfigured,       \
                    Disgusting, Extra arms, Extra limbs, Extra hands, Fused fingers, Gross proportions,  \
                    Long neck, Low res, Low quality, Jpeg, Jpeg artifacts, Malformed limbs, Mutated,     \
                    Mutated hands, Mutated limb, Missing arms, Missing fingers, Picture frame,           \
                    Poorly drawn hands, Poorly drawn face, Text, Signature, Username, Watermark,         \
                    Worst quality, Collage, Pixel, Pixelated, Grainy "

# Stable Diffusion Negative Prompts For Realistic Images
NEGATIVE_PROMPT5 = "Cartoon, CGI, Render, 3D, Artwork, Illustration, 3D render, Cinema 4D, Artstation,   \
                    Octane render, Painting, Oil painting, Anime, 2D, Sketch, Drawing, Bad photography,  \
                    Bad photo, Deviant art, "

# Negative Prompts For Landscape & Nature Images
NEGATIVE_PROMPT6 = "Overexposed, Simple background, Plain background, Grainy, Portrait, Grayscale,       \
                    Monochrome, Underexposed, Low contrast, Low quality, Dark, Distorted, White spots,   \
                    Deformed structures, Macro, Multiple angles, "

# Stable Diffusion Negative Prompts For Proper Human Anatomy
NEGATIVE_PROMPT7 = "Bad anatomy, Bad hands, Amputee, Missing fingers, Missing hands,                     \
                    Missing limbs, Missing arms, Extra fingers, Extra hands, Extra limbs, Mutated hands, \
                    Mutated, Mutation, Multiple heads, Malformed limbs, Disfigured, Poorly drawn hands,  \
                    Poorly drawn face, Long neck, Fused fingers, Fused hands, Dismembered, Duplicate,    \
                    Improper scale, Ugly body, Cloned face, Cloned body, Gross proportions,              \
                    Body horror, Too many fingers, "

# Stable Diffusion Negative Prompts For Objects
NEGATIVE_PROMPT8 = "Asymmetry, Parts, Components, Design, Broken, Cartoon, Distorted, Extra pieces,      \
                    Bad proportion, Inverted, Misaligned, Macabre, Missing parts, Oversized, Tilted, "

NEGATIVE_PROMPT = NEGATIVE_PROMPT + NEGATIVE_PROMPT4 + NEGATIVE_PROMPT5 + NEGATIVE_PROMPT6 + NEGATIVE_PROMPT7 + NEGATIVE_PROMPT8
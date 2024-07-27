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

PROMPT_GENERATE_DESCRIPTION = ""

PROMPT_BRIEF_DESCRIPTION      = "give the brief description of image"
PROMPT_VERB_DESCRIPTION       = "give the description of what to imaging in a line"
PROMPT_IMAGING_DESCRIPTION    = "create the prompt to describe what the image imaging, it will be used   \
                                 for recreating the image excatly"
PROMPT_ONE_LINE_DESCRIPTION   = "Describe the image in one sentence and this description will be used to \
                                 regenerate the same image."
PROMPT_MIDJOURNEY_DESCRIPTION = "Generates prompt that is inspirational and suggestive, it can be used to\
                                 recreate an uploaded image exactly"
PROMPT_BRIEF_MID_DESCRIPTION  = "Create a brief and inspiring imaging prompt that is inspirational and   \
                                 suggestive , which will be used exactly as provided for generating an image from text. The prompt should be in 77 tokens."
PROMPT_DETAILED_DESCRIPTION   = "Carefully analyze the provided image and generate an exhaustive         \
                                 description that captures every element essential for its recreation.   \
                                 Detail the spatial arrangement, key subjects, and any background        \
                                 elements. Specify the color palette, noting any gradients or unique     \
                                 shades present. Describe any actions or interactions between subjects,  \
                                 and capture the mood or atmosphere effectively. Your description should \
                                 be precise and comprehensive, enabling an artist or a generative model  \
                                 to replicate the image with high fidelity based on your text alone."

PROMPT_GENERATE_DESCRIPTION = PROMPT_ONE_LINE_DESCRIPTION

PROMPT_REALISTIC_VISION_POSITIVE = ""

PROMPT_REALISTIC_VISION_NEGATIVE = "deformed iris, deformed pupils, semi-realistic, cgi, 3d, render,     \
                                    sketch, cartoon, drawing, anime), text, cropped, out of frame,       \
                                    worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, \
                                    mutilated, extra fingers, mutated hands, poorly drawn hands,         \
                                    poorly drawn face, mutation, deformed, blurry, dehydrated,           \
                                    bad anatomy, bad proportions, extra limbs, cloned face, disfigured,  \
                                    gross proportions, malformed limbs, missing arms, missing legs,      \
                                    extra arms, extra legs, fused fingers, too many fingers, long neck"
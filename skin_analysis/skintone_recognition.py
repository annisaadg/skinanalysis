import stone

# Skin tone mapping and utilities
skin_tone_to_melanin = {
    "#ffffff": 5,   # Fair Porcelain
    "#a0a0a0": 10,  # Pale Almond
    "#a0a0a0": 20,  # Soft Peach
    "#a0a0a0": 30,  # Creamy Ivory
    "#a0a0a0": 35,  # Light Beige
    "#a0a0a0": 40,  # Beige
    "#a0a0a0": 45,  # Golden Beige
    "#909090": 50,  # Golden Tan
    "#808080": 60,  # Warm Bronze
    "#707070": 65,  # Tan
    "#606060": 70,  # Cocoa Brown
    "#505050": 75,  # Caramel
    "#404040": 80,  # Dark Walnut
    "#303030": 82,  # Deep Amber
    "#202020": 85,  # Rich Chocolate
    "#101010": 90,  # Deep Ebony
    "#000000": 95   # Espresso
}

def get_skin_tone_name(skin_tone):
    skin_tone_names = {
        "#ffffff": "Fair Porcelain", 
        "#f0f0f0": "Pale Almond", 
        "#e0e0e0": "Soft Peach",
        "#d0d0d0": "Creamy Ivory", 
        "#c0c0c0": "Light Beige", 
        "#b0b0b0": "Beige",
        "#a0a0a0": "Golden Beige", 
        "#909090": "Golden Tan", 
        "#808080": "Warm Bronze",
        "#707070": "Tan", 
        "#606060": "Cocoa Brown", 
        "#505050": "Caramel",
        "#404040": "Dark Walnut", 
        "#303030": "Deep Amber", 
        "#202020": "Rich Chocolate",
        "#101010": "Deep Ebony", 
        "#000000": "Espresso"
    }
    return skin_tone_names.get(skin_tone, "Unknown")

class StoneModel:
    def __init__(self):
        pass

    def detect_skin_tone(self, image_path):
        result = stone.process(image_path, image_type="bw", return_report_image=True)
        skin_tone = result["faces"][0]["skin_tone"]
        melanin_level = skin_tone_to_melanin.get(skin_tone, "Unknown")
        skin_tone_name = get_skin_tone_name(skin_tone)

        return {
            "melanin_level": melanin_level,
            "skin_tone_name": skin_tone_name
        },skin_tone
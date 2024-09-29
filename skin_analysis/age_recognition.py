from deepface import DeepFace

# age recognition with deepface
def age_recognition(img_path, skin_core, wrinkles, pigmentation):
    results = DeepFace.analyze( img_path, actions=['age'])
    age = results[0]['age']
    age_result = measure_age(age, skin_core, wrinkles, pigmentation)
    return age_result

# adjust age recognition with skin condition
def measure_age(age, skin_core, wrinkles, pigmentation):
    if (wrinkles == 10):
        w = -2
        sc = -2
        p = -2
    else:
        w = 4 * (10 - wrinkles)
        sc = 2 * (10 - skin_core)
        p = 4 * (10 - pigmentation)
    sum = age + sc + w + p
    return sum
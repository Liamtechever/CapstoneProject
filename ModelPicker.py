from Classifier import Classification
from Model import Model, LocalModel, _available_models, OnlineModel
from Settings import load_settings

# Load settings from settings.json
user_settings = load_settings()

# Now you can use the variables inside ModelPicker
print(f"Using settings: user_vram={user_settings.user_vram}, use_internet={user_settings.use_internet}, api_cost_tolerance={user_settings.api_cost_tolerance}")

# Example usage



#def pick_model(available_models: list[Model], prompt_classification: Classification, settings: Settings, max_number_of_models: int = 1):
def pick_model(available_models: list[Model], prompt_classification: Classification, max_number_of_models: int = 2):
    print(available_models)

    vram_filtered_models = []
    for model in available_models:
        if model.vram <= user_settings.user_vram:
            vram_filtered_models.append(model)

    print(vram_filtered_models)
    api_cost_filtered_models = []


    #print(user_settings.api_cost_tolerance)

    for model in vram_filtered_models:
        if model.cost <= user_settings.api_cost_tolerance:
            api_cost_filtered_models.append(model)

    print("below is the api_cost_filtered models")
    print(api_cost_filtered_models)
    internet_filtered_models = []

    for model in api_cost_filtered_models:
        if not user_settings.use_internet:
            if type(model) == LocalModel:
                internet_filtered_models.append(model)
        elif user_settings.use_internet:
            if type(model) == OnlineModel:
                internet_filtered_models.append(model)


    print("below is the internet filtered models")
    print(internet_filtered_models)
    subject_filtered_models = []

    for model in internet_filtered_models:
        for model_strength in model.strengths:
            if prompt_classification.subject in model_strength.subject_strength:
                subject_filtered_models.append(model)

    print(subject_filtered_models)



    # Sort subject filtered models by strength
    sorted_subject_strength_models = sorted(
        subject_filtered_models,
        key=lambda model: next(
            (strength.strength_level for strength in model.strengths if
             strength.subject_strength == prompt_classification.subject),
            0  # Default to 0 if the subject is not found
        ),
        reverse=True  # Sort from highest to lowest
    )

    print(sorted_subject_strength_models)

    # TODO: Default model if none match

    return sorted_subject_strength_models[:max_number_of_models]

    # TODO: sort subject filtered models by strength


    # TODO: Default model if none match

    #return subject_filtered_models[:max_number_of_models]





if __name__ == '__main__':

    #user_settings = Settings("path/to/settings.json")
   # user_settings = Settings("settings.json")
    user_settings = load_settings()



    #user_settings = Settings(user_vram=500, use_internet=False)
    best_models = pick_model(available_models=_available_models, prompt_classification=Classification("Technology", 3, True))
    #best_models.chat()
    print(best_models)
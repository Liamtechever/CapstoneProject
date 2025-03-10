import json

if __name__ == '__main__':
    #initialize settings if doesn't exist

    internet_response = False
    settings = {"use_internet": internet_response}

    with open("settings.json", "w") as file:
        file.write(json.dumps(settings))

    with open("settings.json", "r") as file:
        settings = json.loads(file.read())

    # load settings

    # Then do AI stuffs





    pass
from django.views.generic import TemplateView
from django.shortcuts import render
from django.utils.safestring import mark_safe

from Model import Model, LocalModel, _available_models, OnlineModel, deepseek_r1_1_5, gpt4_vision, o3mini, gptturbo


from Classifier import classify_prompt
from ModelPicker import pick_model


# Placeholder generate_response function
def generate_response(message):
    # TODO: Load settings

    print(f"Prompt: {message}")

    # Classify the prompt
    prompt_classification = classify_prompt(prompt=message)
    print("Using this classification: ", prompt_classification)



    best_models = pick_model(available_models=_available_models, prompt_classification=prompt_classification)
    base_model = gptturbo
    model = best_models[0]


    response = base_model.chat(prompt=message)

    output = model.chat(prompt=("Below is the original prompt" + message + "Below is the old models response" + response),
               system_message="YOU ARE AN AI THAT ANALYZES OUTPUTS FROM OTHER AI. Only output the improved response. Keep the response to a maximum of three sentences")

    return f"{output}", model.name

class ChatView(TemplateView):
    template_name = 'chat/chat.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['messages'] = self.request.session.get('messages', [])
        return context

    def post(self, request, *args, **kwargs):
        message = request.POST.get('message')

        print(f"Message received: {message}")
        if message:
            messages = request.session.get('messages', [])
            messages.append({'user': True, 'text': message})

            # Expecting generate_response to return (response_text, model_name)
            response_text, model_name = generate_response(message)
            messages.append({
                'user': False,
                'text': response_text,
                'model': model_name
            })

            request.session['messages'] = messages
        return self.get(request, *args, **kwargs)


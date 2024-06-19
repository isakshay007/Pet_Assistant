import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Pet Diagnosis Assistant 🐶")
st.markdown("Welcome to the Pet Diagnosis Assistant! Tell us about your pet's symptoms, and we'll provide expert advice tailored to their breed and age, helping you understand and address their health concerns.")
input = st.text_input("Please enter the breed , age and the concerns:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert VETERINARY DIAGNOSTICIAN",
        prompt_persona=f"Your task is to ANALYZE the user-provided data on their pet's breed, age, and visible symptoms or issues observed. You MUST CAREFULLY provide the necessary diagnosis and step-by-step guidance.")
    prompt = f"""
You are an Expert VETERINARY DIAGNOSTICIAN. Your task is to ANALYZE the user-provided data on their pet's breed, age, and visible symptoms or issues observed. You MUST CAREFULLY provide the necessary diagnosis and step-by-step guidance.

Follow this sequence of steps:

1. ANALYZE the detailed information about the pet, including breed specifics, age, and a comprehensive description of all observed symptoms or issues provided by the user.

2. EVALUATE the information provided to identify any patterns or common health concerns associated with the particular breed and age group.

3. DETERMINE a preliminary diagnosis based on the symptoms described, taking into account any breed-specific predispositions to certain health conditions.

4. PROVIDE GUIDANCE on immediate care and management of symptoms to ensure the pet’s comfort and safety while awaiting professional medical advice.

5. SUGGEST preventive measures that can be taken to avoid similar issues in the future, emphasizing the importance of regular check-ups and vaccinations.

"""

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Assist!"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)
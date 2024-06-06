import streamlit as st

from data import CharTokenizer
from model import Generator


st.title("Text Generator :)")


@st.cache_resource
def load_generator_and_tokenizer():
    # tokenizer = CharTokenizer.load('tokenizer.json')
    tokenizer = CharTokenizer.load('tokenizer.json')

    generator = Generator.load_from_checkpoint("checkpoints/best-checkpoint.ckpt",
                                               tokenizer=tokenizer)

    return tokenizer, generator

tokenizer, generator = load_generator_and_tokenizer()

output = None

with st.form("Generate_Form"):
    st.text("Enter Your Prompt and number of character you want to generate")
    prompt = st.text_input("Prompt.....")
    # n_tokens = st.number_input('n_tokens.....')
    n_tokens = st.number_input('n_tokens.....', value=200)
    # n_tokens = st.number_input('n_tokens.....', min_value=1, step=1)

    
    should_generate = st.form_submit_button('Generate', type='primary', use_container_width=True)
    
    if should_generate:
        with st.spinner("Generating...."):
            output = generator.generate(prompt.lower(),n_tokens)
        

if output is not None:
    st.write(f':red[{prompt}] :blue[{output.replace("`","")}]')
        
        
        
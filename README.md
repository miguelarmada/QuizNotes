# QuizNotes

## Overview

Welcome to the QuizNotes project! This GitHub repository contains a tool that leverages Hugging Face models, LangChain, Streamlit, and FAISS to generate quizzes based on imported class notes. This project aims to simplify the process of creating interactive and engaging quizzes for educational purposes.

## Features

- **Hugging Face Models:** The project utilizes state-of-the-art natural language processing models from Hugging Face, allowing for sophisticated question generation based on class notes.

- **LangChain Integration:** LangChain is integrated to process and understand the language in class notes, enabling the generation of contextually relevant quiz questions.

- **Streamlit UI:** The user interface is built using Streamlit, providing a user-friendly experience for importing class notes, configuring quiz parameters, and generating quizzes.

- **FAISS Indexing:** FAISS is employed for efficient similarity search, enhancing the speed of question generation by quickly identifying relevant sections in the class notes.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/miguelarmada/QuizNotes.git
   ```

2. Change directory:
   ```bash
   cd QuizNotes
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the web interface by opening your browser and navigating to [http://localhost:8501](http://localhost:8501).

3. Follow the instructions on the interface to import your class notes and generate quizzes.

## Contributing

Contributions are welcome! If you have ideas for improvements, bug fixes, or new features, please open an issue or submit a pull request.

Happy quizzing! 📚🎓
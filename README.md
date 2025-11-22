# How to Run This Demo

1. **Install uv**  
    Install uv (a fast Python package installer) by following the instructions at [uv documentation](https://github.com/astral-sh/uv).

2. **Set Up Virtual Environment**  
    Create a virtual environment using uv:
    ```bash
    uv venv
    ```

3. **Install Dependencies**  
    Install project dependencies with:
    ```bash
    uv sync
    ```

4. **Run the Demo**  
    If you want to run the demo, first activate the virtual environment:
    ```bash
    source .venv/bin/activate
    ```

    Or in Windows:
    ```bash
    .venv\Scripts\activate.bat
    ```

    Then start the streamlit app:
    ```bash
    streamlit run app.py
    ```

5. **Run the API**  
    If you just want to run the API, do this:
    ```bash
    uv run api.py
    ```

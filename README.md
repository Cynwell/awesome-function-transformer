# Awesome Function Transformer

This project transforms and visualizes mathematical functions using a Dash web application.  
Designed for:
- **Teachers:** Conduct lessons on function transformations.
- **Students:** Visualize and interact with transformations to build intuition.  

Distinct features from GeoGebra and Desmos:
- Multiple function transformations with a single click.
- Records transformation steps, expressions, and exact transformed equations.
- Self-paced practice and verification of concepts.

## Files in This Repository
- **main.py:** Main application file to run the Dash server.
- **assets/copy.js:** JavaScript asset used by the application.
- **assets/github-mark.svg:** GitHub logo used in the application.

## Setup Instructions
1. **Create the Conda Environment:**
  - ```sh
    conda create -n function-transformer python=3.12 -y
    ```
2. **Activate the Environment:**
  - ```sh
    conda activate function-transformer
    ```
3. **Install Dependencies:**
  - Packages: dash, plotly, numpy, sympy
  - ```sh
    pip install dash plotly numpy sympy
    ```
4. **Run the Application:**
  - ```sh
    python main.py
    ```
5. **View the Application:**
  - Open your browser and navigate to [http://localhost:8050](http://localhost:8050)

---

Enjoy exploring function transformations!
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import sympy as sp
import re

# Define additional trigonometric functions
def csc(x):
    return 1/np.sin(x)
def sec(x):
    return 1/np.cos(x)
def cot(x):
    return 1/np.tan(x)
def csch(x):
    return 1/np.sinh(x)
def sech(x):
    return 1/np.cosh(x)
def coth(x):
    return 1/np.tanh(x)

app = dash.Dash(__name__)
app.title = "Awesome Function Transformer"

# Compose the affine transformation sequentially.
# Initial transform: x' = A*x + B, y' = C*f(x) + D.
def compose_transform(ops):
    A, B, C, D = 1, 0, 1, 0
    for op in ops:
        if op["type"] == "horizontal":
            # f(x+h) shifts the graph left when h>0.
            B += op["value"]
        elif op["type"] == "vertical":
            D += op["value"]
        elif op["type"] == "reflect_y":
            A = -A
            B = -B
        elif op["type"] == "reflect_x":
            C = -C
            D = -D
    return A, B, C, D

# Helper to invert horizontal slider reading.
def invert_horizontal(value):
    return -value

# TODO: `eq_str` can be deleted too
def get_transformed_equation(hor_scale, hor_trans, ver_scale, ver_trans):
    """
    Returns a human‐readable transformed equation string in the form of
    y = [ver_scale]*f((x [± hor_trans])/[hor_scale]) [± ver_trans].
    This function does not expand the base function.
    """
    # Build inner argument.
    if hor_trans == 0:
        inner = "x"
    elif hor_trans > 0:
        inner = f"x + {hor_trans}"
    else:
        inner = f"x - {abs(hor_trans)}"

    # Use a more conventional form when hor_scale == -1.
    if hor_scale == -1:
        inner = f"-{inner}"
    elif hor_scale != 1:
        inner = f"({inner})/{hor_scale}"
    
    # Build the f(...) part.
    f_part = f"f({inner})"
    
    # Build the vertical scaling.
    if ver_scale == 1:
        vs = ""
    elif ver_scale == -1:
        vs = "-"
    else:
        vs = f"{ver_scale}*"
    
    # Build the vertical translation.
    if ver_trans > 0:
        vt = f" + {ver_trans}"
    elif ver_trans < 0:
        vt = f" - {abs(ver_trans)}"
    else:
        vt = ""
    
    return f"{vs}{f_part}{vt}"

def get_exact_transformed_equation(eq_str, hor_scale, hor_trans, ver_scale, ver_trans):
    """
    Returns the exact transformed equation obtained by substitution.
    Uses the substitution: x -> (x + hor_trans)/hor_scale.
    Converts the NumPy-friendly equation into a Sympy-friendly string first,
    passing in correct locals so that e and pi are interpreted properly,
    and then converts exp(...) to the more conventional notation.
    """
    x = sp.symbols('x')
    # Remove "np." so that e.g. "np.sin(x)" becomes "sin(x)"
    sympy_eq_str = eq_str.replace("np.", "")
    # Define a dictionary for sympy to recognize functions and constants.
    local_dict = {
        'sin': sp.sin,
        'cos': sp.cos,
        'tan': sp.tan,
        'log': sp.log,
        'exp': sp.exp,
        'sqrt': sp.sqrt,
        'pi': sp.pi,
        'e': sp.E
    }
    try:
        expr = sp.sympify(sympy_eq_str, locals=local_dict, evaluate=False)
    except Exception:
        return ""
    new_expr = expr.subs(x, (x + hor_trans) / hor_scale) if hor_scale != 0 else expr
    final_expr = ver_scale * new_expr + ver_trans
    # Get string representation.
    final_str = sp.sstr(final_expr)
    # Convert exp( ... ) to the more math-friendly notation e**( ... )
    final_str = final_str.replace("exp(", "e**(")
    # Remove unwanted '*' between a digit and x; e.g., change "3*x" to "3x"
    final_str = re.sub(r'(\d)\*x', r'\1x', final_str)
    return final_str

def process_for_latex_display(eq_str):
    # Replace ** by ^
    eq_str = eq_str.replace("**", "^")
    # Remove '*' between a digit and a letter (e.g., 2*x -> 2x)
    eq_str = re.sub(r'(\d)\*([a-zA-Z])', r'\1\2', eq_str)
    # Replace exponent expressions with parentheses: e^(3x) -> e^{3x}
    eq_str = re.sub(r'\^\(([^\)]+)\)', r'^{\1}', eq_str)
    # Leave exponents without parentheses unchanged (e.g., e^3x remains e^3x)
    # Convert np.pi (or standalone pi) to LaTeX \pi symbol.
    eq_str = eq_str.replace("np.pi", "\\pi")
    eq_str = re.sub(r'(\d*)pi\b', r'\1\\pi', eq_str, flags=re.IGNORECASE)
    return eq_str

# Regex processing for the equation string.
def process_equation(eq_str):
    processed_eq = eq_str

    # Insert multiplication between a digit and "e^", so "2e^x" becomes "2*e^x"
    processed_eq = re.sub(r'(\d)(?=e\^)', r'\1*', processed_eq, flags=re.IGNORECASE)
    # Insert multiplication between a digit and "x"
    processed_eq = re.sub(r'(\d)(?=x)', r'\1*', processed_eq)
    processed_eq = re.sub(r'(\d)(?=(?:sqrt|sin|cos|tan|log|exp|sinh|cosh|tanh'
                          r'|csc|sec|cot|csch|sech|coth)\b)', r'\1*', processed_eq)
    
    processed_eq = re.sub(r'\b(cosec|cosecant)\b', 'csc', processed_eq, flags=re.IGNORECASE)
    processed_eq = re.sub(r'\bcosech\b', 'csch', processed_eq, flags=re.IGNORECASE)
    
    # Insert multiplication between any letter, digit, or closing parenthesis and a following "e^"
    # so that "2xe^x" becomes "2*x*e^x", but avoid inserting an extra multiplication if already present.
    processed_eq = re.sub(r'([a-zA-Z0-9\)])(?=e\^)', r'\1*', processed_eq, flags=re.IGNORECASE)
    
    # Handle exponential expressions with parenthesized exponents, e.g. e^(x+1)
    processed_eq = re.sub(r'(?i)\be\^\(([^)]+)\)', r'np.exp(\1)', processed_eq)
    # Special-case: handle e^pi (without parentheses) as np.exp(np.pi)
    processed_eq = re.sub(r'(?i)\be\^(pi)\b', r'np.exp(np.pi)', processed_eq)
    # Handle exponential expressions without parentheses for single-letter exponents, e.g. e^x
    processed_eq = re.sub(r'(?i)\be\^(?P<exp>[a-zA-Z])', r'np.exp(\g<exp>)', processed_eq)
    # Fallback: convert any "e(" not already prefixed by np. to np.exp(
    processed_eq = re.sub(r'(?<!np\.)\be\(', 'np.exp(', processed_eq)
    
    # Replace any remaining caret with Python’s exponent operator.
    processed_eq = re.sub(r'\^', '**', processed_eq)
    
    processed_eq = re.sub(r'(?<!np\.)\b(sqrt|sin|cos|tan|log|exp|sinh|cosh|tanh)\s*(?=\()', r'np.\1', processed_eq)
    processed_eq = re.sub(r'\|([^|]+)\|', r'np.abs(\1)', processed_eq)
    
    # Insert multiplication between a number and "pi" (e.g., "2pi" -> "2*pi")
    processed_eq = re.sub(r'(\d)(?=pi\b)', r'\1*', processed_eq, flags=re.IGNORECASE)
    processed_eq = re.sub(r'(?i)(?<!np\.)\bpi\b', r'np.pi', processed_eq)
    processed_eq = re.sub(r'(?<![.\w])e(?![.\w])', r'np.e', processed_eq)
    
    # Insert multiplication between a closing parenthesis and an immediately following letter or '('
    processed_eq = re.sub(r'(\))(?=[A-Za-z(])', r'\1*', processed_eq)
    return processed_eq

def base_function(x_data, eq_str):
    local_env = {"np": np, "x": x_data, "csc": csc, "sec": sec, "cot": cot,
                 "csch": csch, "sech": sech, "coth": coth}
    try:
        compiled_expr = compile(eq_str, '<string>', 'eval')
        y_data = eval(compiled_expr, local_env)
        if np.isscalar(y_data):
            y_data = np.full(x_data.shape, y_data)
    except Exception:
        y_data = np.full(x_data.shape, np.nan)
    threshold = 1e6
    y_data = np.where(np.abs(y_data) > threshold, np.nan, y_data)
    return y_data

# Build transformation steps table.
def build_steps_table(ops, eq_str):
    processed_eq = process_equation(eq_str)
    A, B, C, D = 1, 0, 1, 0
    rows = []
    for i, op in enumerate(ops, start=1):
        if op["type"] == "horizontal":
            B += op["value"]
            desc = f"Translate horizontally by {op['value']} unit(s)"
        elif op["type"] == "vertical":
            D += op["value"]
            desc = f"Translate vertically by {op['value']} unit(s)"
        elif op["type"] == "reflect_y":
            A = -A
            B = -B
            desc = "Reflect along y-axis"
        elif op["type"] == "reflect_x":
            C = -C
            D = -D
            desc = "Reflect along x-axis"
        word_eq = get_transformed_equation(A, B, C, D)
        exact_eq = get_exact_transformed_equation(processed_eq, A, B, C, D)
        # Convert the exact_eq into a LaTeX-friendly display version.
        display_eq = process_for_latex_display(exact_eq)
        rows.append(html.Tr([
            html.Td(str(i)),
            html.Td(desc),
            html.Td(dcc.Markdown(f"$$y = {word_eq}$$", mathjax=True)),
            html.Td(dcc.Markdown(f"$$y = {display_eq}$$", mathjax=True)),
            html.Td(html.Button("Copy", id=f"copy_exact_{i}", n_clicks=0,
                            **{"data-equation": exact_eq}, style={"marginLeft": "5px"}))
        ]))
    table = html.Table(
        [html.Thead(html.Tr([
            html.Th("Step"),
            html.Th("Transformation"),
            html.Th("Resulting Equation"),
            html.Th("Exact Equation"),
            html.Th("Copy")
        ]))] +
        [html.Tbody(rows)],
        style={"margin": "auto", "border": "1px solid #ccc", "borderCollapse": "collapse"}
    )
    # Wrap the table in a scrollable div if there are many steps:
    return html.Div(table, style={"maxHeight": "300px", "overflowY": "scroll", "margin": "20px auto", "width": "80%"})

# Updated layout.
app.layout = html.Div([
    dcc.Store(id="stored_eq", data="x**2"),
    dcc.Store(id="ops_store", data=[]),
    html.H1("Awesome Function Transformer", style={"textAlign": "center", "color": "#2c3e50"}),
    html.Div([
        html.Label("Enter Equation: y = ", style={"fontSize": "20px"}),
        dcc.Input(
            id="equation_input",
            type="text",
            value="x**2",
            debounce=True,
            style={"width": "300px", "padding": "10px", "fontSize": "18px"}
        )
    ], style={'margin': '20px', "textAlign": "center"}),
    dcc.Graph(id='graph'),
    html.Div(id="transformation_table", style={"margin": "20px", "textAlign": "center"}),
    html.Div([
        html.Button("Reflect Along X-axis", id="reflect_x", n_clicks=0,
                    style={"fontSize": "16px", "padding": "10px", "marginRight": "10px"}),
        html.Button("Reflect Along Y-axis", id="reflect_y", n_clicks=0,
                    style={"fontSize": "16px", "padding": "10px"})
    ], style={"textAlign": "center", "margin": "20px"}),
    html.Div([
        html.Label("Horizontal Translation (⬅ (+) Shift Left | Shift Right (-) ➡)", style={"fontSize": "18px"}),
        dcc.Slider(
            id='x_shift',
            min=-10, max=10, step=1, value=0,
            marks={i: ('+' if -i > 0 else "") + str(-i) for i in range(-10, 11)},
            tooltip={"always_visible": False}
        )
    ], style={'margin': '20px'}),
    html.Div([
        html.Label("Vertical Translation (⬇ (-) Shift Down | Shift Up (+) ⬆)", style={"fontSize": "18px"}),
        dcc.Slider(
            id='y_shift',
            min=-50, max=50, step=1, value=0,
            marks={i: ('+' if i > 0 else "") + str(i) for i in range(-50, 51, 10)},
            tooltip={"always_visible": False}
        )
    ], style={'margin': '20px'}),
    html.Div([
        html.Button("Reset", id="reset_button", n_clicks=0,
                    style={"fontSize": "16px", "padding": "10px"})
    ], style={"textAlign": "center", "margin": "20px"}),
    html.Footer(
        html.Div([
            html.Span("Developed by Cynwell Lau",
                      style={
                          "fontSize": "16px",
                          "color": "#7f8c8d",
                          "marginRight": "8px",
                          "fontFamily": "Arial, sans-serif"
                      }),
            html.A(
                html.Img(src="assets/github-mark.svg", style={"height": "20px"}),
                href="http://github.com/Cynwell/Awesome-Function-Transformer",
                target="_blank",
                style={"textDecoration": "none"}
            )
        ], style={"display": "flex", "justifyContent": "center", "alignItems": "center"}),
        style={"marginTop": "40px", "padding": "20px"}
    )
], style={"fontFamily": "Arial, sans-serif", "backgroundColor": "#ecf0f1", "padding": "10px"})

x_orig = np.linspace(-10, 10, 5000)

# Combined callback for sliders, reflection buttons, reset, and new equation input.
@app.callback(
    [Output('ops_store', 'data'),
     Output('x_shift', 'value'),
     Output('y_shift', 'value')],
    [Input('x_shift', 'value'),
     Input('y_shift', 'value'),
     Input('reflect_x', 'n_clicks'),
     Input('reflect_y', 'n_clicks'),
     Input('reset_button', 'n_clicks'),
     Input('equation_input', 'n_submit')],
    State('ops_store', 'data'),
)
def update_operations(x_shift, y_shift, rx_clicks, ry_clicks, reset_clicks, n_submit, ops):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]['prop_id']
    if trigger.startswith("equation_input"):
        return [], 0, 0
    ops = ops.copy() if ops else []
    if trigger.startswith("reset_button"):
        return [], 0, 0
    elif trigger.startswith("x_shift"):
        if abs(x_shift) > 1e-8:
            ops.append({"type": "horizontal", "value": invert_horizontal(x_shift)})
        return ops, 0, dash.no_update
    elif trigger.startswith("y_shift"):
        if abs(y_shift) > 1e-8:
            ops.append({"type": "vertical", "value": y_shift})
        return ops, dash.no_update, 0
    elif trigger.startswith("reflect_x"):
        ops.append({"type": "reflect_x"})
        return ops, dash.no_update, dash.no_update
    elif trigger.startswith("reflect_y"):
        ops.append({"type": "reflect_y"})
        return ops, dash.no_update, dash.no_update
    return dash.no_update, dash.no_update, dash.no_update

# Main callback: update graph and transformation table.
@app.callback(
    [Output('graph', 'figure'),
     Output('equation_input', 'value'),
     Output('stored_eq', 'data'),
     Output('transformation_table', 'children')],
    [Input('equation_input', 'n_submit'),
     Input('ops_store', 'data')],
    [State('equation_input', 'value'),
     State('stored_eq', 'data')]
)
def update_graph(n_submit, ops, eq_value, stored_eq):
    if eq_value.strip() != stored_eq:
        ops = []
    current_eq = eq_value if eq_value.strip() != "" else stored_eq
    processed_eq = process_equation(current_eq)
    # Compose transform sequentially.
    x_A, x_B, y_C, y_D = compose_transform(ops)
    final_eq = get_transformed_equation(x_A, x_B, y_C, y_D)
    # Compute original function.
    y_orig = base_function(x_orig, processed_eq)
    # Compute transformed coordinates.
    x_transformed = x_A * x_orig + invert_horizontal(x_B)
    y_transformed = y_C * base_function(x_orig, processed_eq) + y_D
    orig_annotation_text = f"Original Equation: y = f(x) = {current_eq}"
    exact_eq = get_exact_transformed_equation(processed_eq, x_A, x_B, y_C, y_D)
    trans_annotation_text = f"Transformed Equation: y = {final_eq} = {exact_eq}"
    steps_table = build_steps_table(ops, current_eq)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_transformed, y=y_transformed,
                             mode='lines', name='Transformed Function'))
    fig.add_trace(go.Scatter(x=x_orig, y=y_orig,
                             mode='lines', name='Original Function',
                             line=dict(dash="dash")))
    fig.update_layout(
        title="Function Transformation Visualization",
        template="plotly_white",
        xaxis=dict(range=[-12, 12], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        yaxis=dict(range=[-60, 60], zeroline=True, zerolinewidth=2, zerolinecolor='black'),
        annotations=[
            dict(x=0.5, y=1.21, xref='paper', yref='paper',
                 text=orig_annotation_text, showarrow=False, font=dict(size=14)),
            dict(x=0.5, y=1.15, xref='paper', yref='paper',
                 text=trans_annotation_text, showarrow=False, font=dict(size=14)),
            dict(x=12, y=0, xref='x', yref='y', text="x", showarrow=False,
                 xanchor="right", yanchor="top", font=dict(size=14)),
            dict(x=0, y=60, xref='x', yref='y', text="y", showarrow=False,
                 xanchor="left", yanchor="bottom", font=dict(size=14))
        ]
    )
    sample_indices = [int(0.3*len(x_orig)), int(0.4*len(x_orig)),
                      int(0.5*len(x_orig)), int(0.6*len(x_orig)), int(0.7*len(x_orig))]
    for i in sample_indices:
        fig.add_annotation(
            x=x_transformed[i], y=y_transformed[i],
            ax=x_orig[i], ay=y_orig[i],
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=3, arrowsize=2,
            arrowwidth=2, arrowcolor="blue", text=""
        )
    return fig, current_eq, current_eq, steps_table

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)
import pandas as pd

# Load the data from the Excel file
file_path = './results/results_100_50.xlsx'
data = pd.read_excel(file_path)

# Convert column names to uppercase
data.columns = map(str.upper, data.columns)

# Create a function to generate LaTeX table
def generate_latex_table(df):
    # Begin the LaTeX table
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{adjustbox}{width=\\textwidth}\n\\begin{tabular}{l" + "r" * (len(df.columns) - 1) + "}\n"
    
    # Add the table header
    header = " & ".join(df.columns) + " \\\\\n"
    latex_table += header + "\\hline\n"
    
    # Add the table rows
    for index, row in df.iterrows():
        row_values = row.values
        numeric_values = row_values[1:]  # Exclude the first column for min calculation
        min_value = min(numeric_values)  # Find the minimum numeric value
        
        row_str = row['UNNAMED: 0']  # Start the row string with the first column value
        for value in row_values[1:]:
            if value == min_value:
                row_str += f" & \\textbf{{{value:.3e}}}"
            else:
                row_str += f" & {value:.3e}"
        latex_table += row_str + " \\\\\n"
    
    # End the LaTeX table
    latex_table += "\\end{tabular}\n\\end{adjustbox}\n\\caption{Results from the experiments}\n\\label{tab:results}\n\\end{table}"
    
    return latex_table

# Generate LaTeX table from the dataframe
latex_code = generate_latex_table(data)

# Save the LaTeX code to a file
with open('./results/results_100_50.tex', 'w') as file:
    file.write(latex_code)

print(latex_code)

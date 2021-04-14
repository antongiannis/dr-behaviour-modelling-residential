if (!require(collapsibleTree)) install.packages('collapsibleTree')
if (!require(htmlwidgets)) install.packages('htmlwidgets')
library(collapsibleTree)
library(htmlwidgets)

# Read dataframe created by the python script
csv_path <- file.path(getwd(), "data", "structure_df.csv")
dataframe_structure <- read.csv(csv_path)

# Create the tree graph
tree_structure <- collapsibleTree(
  dataframe_structure,
  fill = "#64ABC2",
  hierarchy = c("columns", "col_values"),
  width = 1000,
  height = 1500,
  zoomable = FALSE
)

# Create directory if it doesn't exist
dir.create("output_graphs", showWarnings = FALSE)

# Write it to a html file
withr::with_dir('output_graphs', saveWidget(tree_structure, file="tree_structure.html"))
print("Structure collapsible tree created successfully!")
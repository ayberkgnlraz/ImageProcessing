# Load required libraries
library(tidyr)
library(dplyr)

# Copy paste the output we got
object_dimensions <- c(
  "Object 1: 46.99 mm x 69.61 mm",
  "Object 2: 63.23 mm x 93.97 mm",
  "Object 3: 50.18 mm x 33.93 mm",
  "Object 4: 44.38 mm x 66.13 mm",
  "Object 5: 66.13 mm x 44.67 mm",
  "Object 6: 44.67 mm x 66.42 mm",
  "Object 7: 33.65 mm x 50.18 mm",
  "Object 8: 44.38 mm x 66.42 mm",
  "Object 9: 62.94 mm x 93.97 mm"
)

# Creating a data frame
object_dimensions_df <- data.frame(
  object = integer(length(object_dimensions)),
  height = numeric(length(object_dimensions)),
  length = numeric(length(object_dimensions)),
  stringsAsFactors = FALSE
)

# Extracting data from the character vector
object_dimensions_df <- object_dimensions_df %>%
  mutate(
    object = seq_along(object_dimensions),
    height = as.numeric(sub("Object \\d+: ([0-9.]+) mm x [0-9.]+ mm", "\\1", object_dimensions)),
    length = as.numeric(sub("Object \\d+: [0-9.]+ mm x ([0-9.]+) mm", "\\1", object_dimensions))
  )

# Printing the resulting data frame
print(object_dimensions_df)

#Visualization part

library(ggplot2)

ggplot(object_dimensions_df, aes(x = length, y = height, color = as.factor(object))) +
  geom_point(size = 3) +  # Adjust the size as needed
  labs(title = "Scatter Plot of Rectangles Dimensions",
       x = "Length (mm)",
       y = "Height (mm)")




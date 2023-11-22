library(data.table)
library(ggplot2)
library(scales)

# read processed data: 17th hour of January 18, 2023
df <- fread("ignore/data_train/processed_subset/20230118_17.txt")
df[, Parameter := factor(Parameter, levels = rev(c(9, 27, 10, 28, 11, 29, 6)))]
df[, SCA := factor(SCA, levels = c('GG', 'GY', 'YY', 'YR', 'YG', 'RR', 'RG', 'XX'))]

SCA_colors <- c(
    'GG' = 'forestgreen',
    'GY' = 'limegreen', 
    'YY' = 'orange', 
    'YR' = 'brown', 
    'YG' = 'blue', 
    'RR' = 'red', 
    'RG' = 'black', 
    'XX' = 'purple'
)

# minute 5 & 6
df5 <- copy(df)[minute(TimeStamp) %in% c(5, 6)]
df5[, example := '(c) Partial inference']

# minute 8
df8 <- copy(df)[minute(TimeStamp) == 8]
df8[, example := '(b) Full inference']

# minute 15 & 16
df15 <- copy(df)[minute(TimeStamp) %in% c(15, 16)]
df15[, example := '(d) Partial inference']

# minute 25
df25 <- copy(df)[minute(TimeStamp) == 25]
df25[, example := '(a) Full inference']

# row bind all datasets
mdf <- rbind(df5, df8, df15, df25)
mdf <- mdf[Parameter != 6, ]

plot <- ggplot(mdf, aes(x = TimeStamp, y = Parameter, color = SCA)) + 
    geom_point(size = 3) + 
    facet_wrap(~example, scale = 'free_x', nrow = 1) + 
    scale_color_manual(values = SCA_colors) +
    scale_x_datetime(labels = scales::date_format('%M:%S')) + 
    xlab("Timestamp of detector actuations at 5 pm (January 18, 2023)") + 
    ylab("Detector ID") + 
    theme_minimal() + 
    theme(axis.text.x = element_text(size = 12),
          axis.text.y = element_text(size = 12),
          axis.title = element_text(size = 14, face = 'bold'),
          legend.title = element_text(size = 14, face = 'bold'),
          legend.text = element_text(size = 12),
          legend.background = element_rect(color = 'black'),
          legend.position = 'top',
          legend.box = 'vertical',
          legend.direction = 'horizontal',
          legend.spacing = unit(0.5, 'cm'),
          legend.box.margin = margin(0, 0, 0, 0, 'cm'),
          panel.border = element_rect(color = 'black', fill = NA),
          panel.background = element_rect(color = 'NA'),
          strip.text = element_text(size = 12, face = 'bold'),
          plot.background = element_rect(fill = 'white', color = 'NA')) + 
    guides(color = guide_legend(title = 'SCA', override.aes = list(size = 3)))

plot

ggsave("output/static_plots/example_infer_reid_pairs.png",
       plot = plot,
       units = 'cm',
       width = 29.7,
       height = 18,
       dpi = 1200)

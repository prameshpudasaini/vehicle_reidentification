library(data.table)
library(ggplot2)

df <- fread("ignore/predicted_travel_time_all_models.txt", sep = '\t')

models <- c('SVR', 'RF', 'XGB', 'FCNN', 'TabNet', 'NODE')
groups <- c('True Positive', 'False Negative')

df[, Model := factor(Model, levels = models)]
df[, Group := factor(Group, levels = groups)]

# compute correlation
compute_corr <- function(model){
    temp <- copy(df)[df$Model == model]
    corr <- round(cor(temp$travel_time, temp$y_pred), 2)
    return(corr)
}

cor_svr <- compute_corr('SVR')
cor_rf <- compute_corr('RF')
cor_xgb <- compute_corr('XGB')
cor_fcnn <- compute_corr('FCNN')
cor_tabnet <- compute_corr('TabNet')
cor_node <- compute_corr('NODE')

df_cor <- as.data.table(data.frame(
    label = paste0("r = ", c(cor_svr, cor_rf, cor_xgb, cor_fcnn, cor_tabnet, cor_node)),
    Model = models
))
df_cor[, Model := factor(Model, levels = models)]

# min max travel times
tt_min <- min(df$travel_time)
tt_max <- max(df$travel_time)    
pred_min <- min(df$y_pred)
pred_max <- max(df$y_pred)

# scatter plot: ground-truth vs predicted travel times
plot_tt_comparison <- ggplot(df, aes(x = travel_time, y = y_pred)) + 
    geom_point(aes(color = Group), shape = 1, size = 1.5) + 
    facet_wrap(~Model) + 
    geom_text(data = df_cor, aes(4.1, 10.5, label = label), size = 5) +
    scale_x_continuous(breaks = seq(3, 11, 2)) + 
    scale_y_continuous(breaks = seq(3, 11, 2)) + 
    scale_color_manual(values = c('forestgreen', 'red')) + 
    xlab("Ground-truth travel time (sec)") + 
    ylab("Predicted travel time (sec)") + 
    theme_minimal() + 
    theme(axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14),
          axis.title = element_text(size = 16, face = 'bold'),
          legend.title = element_blank(),
          legend.text = element_text(size = 14),
          legend.position = 'top',
          legend.box = 'vertical',
          legend.spacing = unit(0, 'cm'),
          legend.box.margin = margin(0, 0, 0, 0, 'cm'),
          panel.border = element_rect(color = 'black', fill = NA),
          panel.background = element_rect(color = 'NA'),
          strip.text = element_text(size = 16, face = 'bold'),
          plot.background = element_rect(fill = 'white', color = 'NA')) + 
    guides(shape = guide_legend(override.aes = list(size = 5)),
           color = guide_legend(override.aes = list(size = 5)))
plot_tt_comparison

ggsave("output/static_plots/model_error_scatter_v2.png",
       plot = plot_tt_comparison,
       units = 'cm',
       width = 29.7,
       height = 21,
       dpi = 1200)

# extra plots

df[, Error := travel_time - y_pred]

ggplot(df, aes(x = Model, y = Error)) + 
    geom_violin() + 
    geom_boxplot(width = 0.2, position = position_dodge(0.9))

ggplot(df, aes(x = Model, y = Error, color = Group)) + 
    geom_boxplot(width = 0.75, position = position_dodge(0.9))

ggplot(df, aes(x = Model, y = Error, color = Group)) + 
    geom_violin() + 
    geom_boxplot(width = 0.2, position = position_dodge(0.9))

fn <- copy(df)[Group == 'False Negative', ]
ggplot(fn, aes(x = Error)) + 
    geom_histogram(bins = 8) + 
    facet_wrap(~ Model, ncol = 3)

# box plot of errors in travel time prediction

error_min = min(df$Error)
error_max = max(df$Error)
error_min = -8
error_max = 6

ggplot(df, aes(x = Model, y = Error, color = Group)) + 
    geom_boxplot(width = 0.75, position = position_dodge(0.9)) + 
    scale_color_manual(values = c('forestgreen', 'red')) + 
    scale_y_continuous(breaks = seq(error_min, error_max, 2)) + 
    xlab("") + 
    ylab("Error in travel time prediction (sec)") + 
    labs(color = "") + 
    theme_minimal() + 
    theme(axis.text.x = element_text(size = 14, face = 'bold'),
          axis.title.x = element_blank(),
          axis.text.y = element_text(size = 14),
          axis.title.y = element_text(size = 16, face = 'bold'),
          legend.title = element_text(size = 16, face = 'bold'),
          legend.text = element_text(size = 14),
          legend.position = c(0.75, 0.925),
          legend.box = 'vertical',
          legend.direction = 'horizontal',
          legend.spacing = unit(0.5, 'cm'),
          legend.box.margin = margin(0, 0, 0, 0, 'cm'),
          panel.border = element_rect(color = 'black', fill = NA),
          panel.background = element_rect(color = 'NA'),
          panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          strip.text = element_text(size = 16, face = 'bold'),
          plot.background = element_rect(fill = 'white', color = 'NA')) + 
    guides(shape = guide_legend(override.aes = list(size = 5)),
           color = guide_legend(override.aes = list(size = 5)))

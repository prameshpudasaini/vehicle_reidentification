library(data.table)
library(ggplot2)
library(e1071)

best_match_model <- 'Best Reidentification'
best_pred_model <- 'Best Prediction'
models <- c('Decision Tree Regression', 'Support Vector Regression', 'Random Forest', 'XGBoost')

width <- 29.7
height <- 21
dpi <- 1200

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# error in travel time prediction
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

df_error <- fread("data/result_prediction_error.txt")
df_error[, model := factor(model, levels = models)]

dt_match <- df_error[model == 'Decision Tree Regression']$best_match_error
dt_pred <- df_error[model == 'Decision Tree Regression']$best_pred_error
sv_match <- df_error[model == 'Support Vector Regression']$best_match_error
sv_pred <- df_error[model == 'Support Vector Regression']$best_pred_error
rf_match <- df_error[model == 'Random Forest']$best_match_error
rf_pred <- df_error[model == 'Random Forest']$best_pred_error
xgb_match <- df_error[model == 'XGBoost']$best_match_error
xgb_pred <- df_error[model == 'XGBoost']$best_pred_error

cor_dt <- round(cor(dt_match, dt_pred), 4)
cor_sv <- round(cor(sv_match, sv_pred), 4)
cor_rf <- round(cor(rf_match, rf_pred), 4)
cor_xgb <- round(cor(xgb_match, xgb_pred), 4)

df_cor <- data.frame(
    label = paste0("r = ", c(cor_dt, cor_sv, cor_rf, cor_xgb)),
    model = models
)

plot_error_scatter <- ggplot(df_error, aes(x = best_match_error, y = best_pred_error)) + 
    geom_point(size = 0.5) + 
    facet_wrap(~model) + 
    geom_text(data = df_cor, aes(0, -4, label = label), size = 5) +
    xlab("Error in travel time prediction for best reidentification (sec)") + 
    ylab("Error in travel time prediction for best prediction (sec)") + 
    theme_minimal() + 
    theme(axis.text.x = element_text(size = 14),
          axis.text.y = element_text(size = 14),
          axis.title = element_text(size = 16, face = 'bold'),
          legend.title = element_text(size = 16, face = 'bold'),
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

plot_error_scatter

ggsave("output/static_plots/models_error_scatter.png",
       plot = plot_error_scatter,
       units = 'cm',
       width = width,
       height = height,
       dpi = dpi)

df_error <- melt(
    df_error, 
    id.vars = 'model', 
    measure.vars = c('best_match_error', 'best_pred_error'),
    variable.name = 'type'
)

skew_kurt <- df_error[, .(Mean = round(mean(value), 2),
                          Median = round(median(value), 2),
                          SD = round(sd(value), 2),
                          Skewness = round(skewness(value), 2), 
                          Kurtosis = round(kurtosis(value), 2)), by = .(model, type)]

error_min <- floor(min(df_error$value) / 2) * 2
error_max <- ceiling(max(df_error$value) / 2) * 2

col_match <- 'red'
col_pred <- 'blue'

plot_error_violin <- ggplot(df_error, aes(x = model, y = value, color = type)) + 
    geom_violin() + 
    geom_boxplot(width = 0.2, position = position_dodge(0.9)) + 
    scale_color_manual(values = c(col_match, col_pred), labels = c(best_match_model, best_pred_model)) + 
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
          legend.position = c(0.75, 0.2),
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

plot_error_violin

ggsave("output/static_plots/models_error_violin.png",
       plot = plot_error_violin,
       units = 'cm',
       width = width,
       height = height,
       dpi = dpi)

# KS test for error in prediction across models

dt_match <- df_error[model == 'Decision Tree Regression' & type == 'best_match_error', ]$value
dt_pred <- df_error[model == 'Decision Tree Regression' & type == 'best_pred_error', ]$value
sv_match <- df_error[model == 'Support Vector Regression' & type == 'best_match_error', ]$value
sv_pred <- df_error[model == 'Support Vector Regression' & type == 'best_pred_error', ]$value
rf_match <- df_error[model == 'Random Forest' & type == 'best_match_error', ]$value
rf_pred <- df_error[model == 'Random Forest' & type == 'best_pred_error', ]$value
xgb_match <- df_error[model == 'XGBoost' & type == 'best_match_error', ]$value
xgb_pred <- df_error[model == 'XGBoost' & type == 'best_pred_error', ]$value

ks.test(dt_match, dt_pred)
ks.test(sv_match, sv_pred)
ks.test(rf_match, rf_pred)
ks.test(xgb_match, xgb_pred)

ks.test(dt_match, sv_match)
ks.test(dt_match, rf_match)
ks.test(dt_match, xgb_match)
ks.test(sv_match, rf_match)
ks.test(sv_match, xgb_match)
ks.test(rf_match, xgb_match)

ks.test(dt_pred, sv_pred)
ks.test(dt_pred, rf_pred)
ks.test(dt_pred, xgb_pred)
ks.test(sv_pred, rf_pred)
ks.test(sv_pred, xgb_pred)
ks.test(rf_pred, xgb_pred)

# Vehicle Reidentification between Advance and Stop-bar Detectors

This research focuses on reidentifying vehicles between advance and stop-bar detectors at signalized intersections using high-resolution traffic data. An optimization framework, proposed on top of ML models that predict the travel time from advance to stop-bar locations, is used to evaluate the accuracy of correct match pairs between the two detectors. The ML models are trained using semi-ground-truth data, while the accuracy of the match pairs are tested on video-verified ground-truth data.

## Script structure
1. preprocess_training_data.py
2. process_events.py
3. generate_candidate_matches.py
4. match_pairs_train_dataset.py
5. process_match_pairs.py
6. feature_extraction.py
7. ML_models.py

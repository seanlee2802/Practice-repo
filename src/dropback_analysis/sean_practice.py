def normalize_play_direction(x, play_direction, field_length=120):
    if play_direction == 'left':
        return field_length - x
    return x

print(normalize_play_direction(50, 'left'))


def evaluate_predictions(y_true, y_pred_proba, threshold=0.5):
    y_pred_class = (y_pred_proba >= threshold)
    accuracy = (y_pred_class == y_true).mean()
    return {"accuracy": accuracy}



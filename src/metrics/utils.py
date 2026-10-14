# Based on seminar materials

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:

    last_line = [i for i in range(len(target_text) + 1)]
    current_line = [0] * (len(target_text) + 1)

    for i in range(1, len(predicted_text) + 1):
        current_line[0] = i
        for j in range(1, len(target_text) + 1):
            cost = 0 if predicted_text[i - 1] == target_text[j - 1] else 1
            current_line[j] = min(
                current_line[j - 1] + 1,
                last_line[j] + 1,
                last_line[j - 1] + cost,
            )
        last_line, current_line = current_line, last_line

    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0

    return last_line[len(target_text)] / len(target_text)

    


def calc_wer(target_text, predicted_text) -> float:
    
    target_text = target_text.split()
    predicted_text = predicted_text.split()
    
    last_line = [i for i in range(len(target_text) + 1)]
    current_line = [0] * (len(target_text) + 1)

    for i in range(1, len(predicted_text) + 1):
        current_line[0] = i
        for j in range(1, len(target_text) + 1):
            cost = 0 if predicted_text[i - 1] == target_text[j - 1] else 1
            current_line[j] = min(
                current_line[j - 1] + 1,
                last_line[j] + 1,
                last_line[j - 1] + cost,
            )
        last_line, current_line = current_line, last_line

    if len(target_text) == 0:
        return 1.0 if len(predicted_text) > 0 else 0.0

    return last_line[len(target_text)] / len(target_text)

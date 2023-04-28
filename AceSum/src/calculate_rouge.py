from datasets import load_metric


def calculate(gold_sums, pred_sums):
  metric = load_metric("rouge")

  result = metric.compute(predictions=pred_sums, references=gold_sums, language="ru", use_stemmer=True)
  result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}

  return result['rouge1'], result['rouge2'], result['rougeL']

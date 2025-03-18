# пример использования MetricEvaluator
# на датсете: data/snils_dataset_annotations.csv
# ответах модели: data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv

from metric_evaluator.metric_evaluator import MetricEvaluator

if __name__ == "__main__":
    dataset_annot = "data/snils_dataset_annotations.csv"
    model_answers = (
        "data/snils_MODELFRAMEWORK_Qwen2-VL-2B-Instruct_VQA_answers_20250124_125639.csv"
    )
    metrics_aggregators = [
        "by_id",
        "by_doc_question",
        "by_doc_type",
        "general",
    ]

    metric_eval = MetricEvaluator(dataset_annot, model_answers)

    # Производим расчет метрик по всем агрегаторам
    for metrics_aggregator in metrics_aggregators:
        metric_csv_path = f"workspace/ModelsMetrics/df_{metrics_aggregator}.csv"
        df_by_id = metric_eval.calculate_metrics_by_id()
        df_by_id.to_csv(
            metric_csv_path,
            sep=";",
            encoding="utf-8-sig",
            index=False,
        )
